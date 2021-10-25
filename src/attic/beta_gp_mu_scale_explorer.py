import math
import torch
from torch import distributions as distrs
import numpy as np
import sys
import tqdm
import pickle
import torch.nn.functional as F

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy

from . import dataset, data_fitter
from .utils import config

if config.xla:
    import torch_xla.core.xla_model as xm

EMB_SIZE = 50

    
def mu_transform(x):
    # 0 when \leq -1 and 1 when \geq 1 else x/2 + 1/2
#     return x
    return torch.sigmoid(x)
#     return torch.threshold(-torch.threshold(-x, threshold=-1e-3, value=-1e-3), threshold=1-1e-3, value=1-1e-3)
#     return torch.nn.functional.hardsigmoid(x)

def scale_transform(x):
    # softplus
    if type(x) != torch.Tensor:
        x = torch.tensor(x)
    return torch.nn.functional.softplus(x)
    # return torch.log(1+np.exp(-np.abs(x))) + np.max(x, 0)
    
class BetaBinomialLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    """
    These are the only two functions that are needed for ELBO computation
    """
    def forward(self, function_samples, **kwargs):
        # function_samples would be of shape 20(num samples) x batch_size x 2(num_tasks) 
        mu, scale = mu_transform(function_samples[:, :, 0]), scale_transform(function_samples[:, :, 1])
        count = kwargs["count"]
        alphas, betas = mu*scale, (1-mu)*scale
        return gpytorch.distributions.base_distributions.BetaBinomial(alphas, betas, count)
    
    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        log_prob_lambda = lambda function_samples: self.forward(function_samples, **kwargs).log_prob(observations)
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob
    
    
class BetaClosenessLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    """
    These are the only two functions that are needed for ELBO computation
    """
    def forward(self, function_samples, **kwargs):
        # function_samples would be of shape 20(num samples) x batch_size x 2(num_tasks) 
        mu, scale = mu_transform(function_samples[:, :, 0]), scale_transform(function_samples[:, :, 1])
        count = kwargs["count"]
        def fn(observations):
            alpha, beta = observations, count - observations
            beta_obs = torch.distributions.Beta(alpha, beta)
            obs_mean = beta_obs.mean
#             obs_mean = torch.min(obs_mean, torch.ones_like(obs_mean)*0.95)
#             obs_mean = torch.max(obs_mean, torch.ones_like(obs_mean)*0.05)
            beta_est = torch.distributions.Beta(mu*scale, (1-mu)*scale)
            # num_samples x batch_size
            n1 = torch.distributions.Normal(beta_est.mean, scale=1e-1).log_prob(obs_mean)
            # num_samples x batch_size
            n2 = torch.distributions.Normal(beta_est.variance + (beta_est.mean**2), scale=1e-1).log_prob(beta_obs.variance + (beta_obs.mean**2))
            return n1 + n2 
        return fn
    
    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        log_prob_lambda = lambda function_samples: self.forward(function_samples, **kwargs)(observations)
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob

    
class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x, embedder):
        inducing_points = torch.randn([2, 500, train_x.size(1)])
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([2]))
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ), num_tasks=2
        )
        super().__init__(variational_strategy)
        
#         prior = gpytorch.priors.NormalPrior(loc=torch.tensor([0., -10]), scale=1.)
        prior = gpytorch.priors.UniformPrior(torch.tensor([0., 0.]), torch.tensor([0.5, 0.]))
        self.mean_module = gpytorch.means.ConstantMean(prior=prior, batch_shape=torch.Size([2]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2])),
            batch_shape=torch.Size([2]), 
            ard_num_dims=EMB_SIZE
        )
        self.embedder = embedder

    def forward(self, x):
        embeds = self.embedder(x)
        mean_x = self.mean_module(embeds)
        covar_x = self.covar_module(embeds)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
    
    
class BetaGPExplorer(torch.nn.Module):
    def __init__(self, dataset: dataset.Dataset, fitter: data_fitter.Fitter, cache_dir: str, 
                 device, explore_strategy='variance', err_ns=[.03, .05, .1]):
        super(BetaGPExplorer, self).__init__()
        self.err_ns = err_ns
        self.dataset = dataset
        self.fitter = fitter
        self.cache_dir = cache_dir
        self.dev = device
        self.explore_strategy = explore_strategy
        
        num_arms = len(dataset.arms)

        # initialize alpha, beta params for all the arms
        self.counts0, self.counts1 = torch.zeros([num_arms], device=self.dev), torch.zeros([num_arms], device=self.dev)
        self.num_arms = num_arms
        self.init_state_dict = fitter.kernel_model.state_dict()
        self.num_obs = 0
    
        linear_layer1 = torch.nn.Linear(self.fitter.kernel_model.emb_size, EMB_SIZE)
#         linear_layer2 = torch.nn.Linear(EMB_SIZE, EMB_SIZE)
        self.embedder = torch.nn.Sequential(
            self.fitter.kernel_model.embedding_model,
            torch.nn.ReLU(),
            linear_layer1,
        ).to(self.dev)
        self._init_params()
        self.arms_tensor = torch.from_numpy(self.dataset.arms).type(torch.float32).to(self.dev)

        # Initialize model and likelihood
        self.gp_model = GPClassificationModel(self.arms_tensor, self.embedder).to(self.dev)
        # likelihood = gpytorch.likelihoods.BernoulliLikelihood()
#         self.gp_likelihood = BetaBinomialLikelihood()
        self.gp_likelihood = BetaClosenessLikelihood()

        kernel_params = list(self.embedder.parameters())
        kernel_param_names = [n for n, _ in self.embedder.named_parameters()]
        all_but_kernel_params = [param for n, param in self.gp_model.named_parameters() if n not in kernel_param_names]
        
        self.optimizer = torch.optim.Adam(all_but_kernel_params + kernel_params, lr=1e-3, weight_decay=0)
#         self.optimizer = torch.optim.SGD(list(self.gp_model.parameters()) + list(self.embedder.parameters()), lr=1e-1)

        # for 1D integrals 
        self.quadrature = gpytorch.utils.quadrature.GaussHermiteQuadrature1D()
    
    def _init_params(self):
        for param in self.embedder.parameters():
            torch.nn.init.normal_(param, 0, 1e-3)
        self.fitter.kernel_model.load_state_dict(self.init_state_dict)
        
    def mean_variance(self, debug=False):
        self.gp_model.eval()
        self.embedder.eval()
        self.gp_likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            z = self.gp_model(self.arms_tensor)
            latent_mean, latent_variance = z.mean, z.variance
            normal_dist = distrs.Normal(latent_mean, latent_variance)

            count = self.counts0 + self.counts1

            def mean_fn(sample):
                mu, scale = mu_transform(sample[:, :, 0]), scale_transform(sample[:, :, 1])
                return mu

            def variance_fn(sample):
                mu, scale = mu_transform(sample[:, :, 0]), scale_transform(sample[:, :, 1])
                return mu*(1-mu)/(scale + 1e-5)

            mean = self.quadrature(mean_fn, normal_dist)
            variance = self.quadrature(variance_fn, normal_dist)
            
            with open(self.cache_dir + "/test_beta_gp.pkl", "wb") as f:
                pickle.dump((mean.numpy(), self.counts0.numpy(), self.counts1.numpy()), f)
            if debug:
                return mean, variance, latent_mean[:, 0], latent_mean[:, 1]
            else:
                return mean, variance
    
    def fit(self): 
        self.gp_model.train()
        self.gp_likelihood.train()
        
        obs_idxs = torch.where((self.counts0 + self.counts1) > 0)[0]
        count = (self.counts0 + self.counts1)[obs_idxs]
        obs_y = self.counts1[obs_idxs]
        
        mll = gpytorch.mlls.VariationalELBO(self.gp_likelihood, self.gp_model, obs_y.numel())
        reg_coeff = 1e-2

        training_iterations = 50
        for i in range(training_iterations):
            self.optimizer.zero_grad()
            latent_mean = self.gp_model(self.arms_tensor[obs_idxs]).mean
            reg_loss = reg_coeff*torch.mean(latent_mean[:, 0]**2)
            reg_loss += reg_coeff*torch.mean(latent_mean[:, 1]**2)
#             for param in self.gp_model.parameters():
#                 reg_loss += reg_coeff*torch.norm(param)
        
            arm_embeddings = self.embedder(self.arms_tensor[obs_idxs])
            output = self.gp_model(self.arms_tensor[obs_idxs])
            loss = -mll(output, obs_y, count=count)
            loss += reg_loss
            
            loss.backward()
            self.optimizer.step()
        
        # debug stuff
        mean, variance, a, b = self.mean_variance(debug=True)
        emp_mean = obs_y/count
        errs = np.abs((emp_mean - mean[obs_idxs]).numpy())
        _idx, idx = np.argmax(errs), obs_idxs[np.argmax(errs)]
        
        print ("Worst error: %0.4f counts: %f %f a, b: %0.3f %0.3f" % (errs[_idx], self.counts0[idx], self.counts1[idx], a[idx], b[idx]))
        print ("Median, mean error:", np.median(errs), np.mean(errs))
        print ("")

    def explore(self, num_explore):
        with torch.no_grad():
            mean, variance, a, b = self.mean_variance(debug=True)
            if self.explore_strategy == 'variance':
                arm_idxs = torch.argsort(-variance)
                arm_idxs = arm_idxs.cpu().numpy()
            elif self.explore_strategy == 'inv_count':
                arm_idxs = torch.argsort(-1/b)
                arm_idxs = arm_idxs.cpu().numpy()
            elif self.explore_strategy == 'nmean':
                arm_idxs = torch.argsort(-mean)
                arm_idxs = arm_idxs.cpu().numpy()
            elif self.explore_strategy == 'random':
                arm_idxs = np.arange(self.num_arms)
                np.random.shuffle(arm_idxs) 
            
            x, y = [], []
            obs_arm_idxs = []
#             num_sample = 5
            num_sample = 3
            for arm_idx in tqdm.tqdm(arm_idxs, desc="sampling..."):
                status = self.fitter.sample_arm(arm_idx, num_sample=num_sample, counts0=self.counts0, counts1=self.counts1)
                if status:
                    obs_arm_idxs += [arm_idx]*num_sample
                if len(obs_arm_idxs) >= num_explore:
                    break

            _c0, _c1 = self.counts0.cpu().numpy(), self.counts1.cpu().numpy()
            sys.stderr.write("Err of arms being explored: %s\n" % str([(ai, _c0[ai], _c1[ai], a[ai], b[ai], variance[ai]) for ai in obs_arm_idxs]))
            _s1, _s2 =  np.sort(a.detach().numpy()), np.sort(b.detach().numpy())
            sys.stderr.write("%s %s\n" % (_s1[-10:], _s1[:10]))
            sys.stderr.write("%s %s\n" % (_s2[-10:], _s2[:10]))
            
    def eval(self):
        mean, variance = self.mean_variance()
        mu_hat = mean.detach().cpu().numpy()
        err_1, all_err = self.fitter.evaluate_mean_estimates(mu_hat, self.err_ns[0], debug=True)
        err_2, err_3 = [self.fitter.evaluate_mean_estimates(mu_hat, _)[0] for _ in self.err_ns[1:]]
        
        bad_arms = np.argsort(-np.array(all_err))
        _c0, _c1 = self.counts0.cpu().numpy(), self.counts1.cpu().numpy()
        
        debug_obj = [(ai, all_err[ai], self.fitter.arm_index_to_acc[ai], _c0[ai], _c1[ai]) for ai in bad_arms]
        with open("%s/debug_beta_gp_mu_scale.pkl" % self.cache_dir, "wb") as f:
            pickle.dump(debug_obj, f)
        
        debug_str = str(["%d %0.4f %0.4f %0.3f %0.3f" % (ai, all_err[ai], self.fitter.arm_index_to_acc[ai], _c0[ai], _c1[ai]) for ai in bad_arms[:50]])
        sys.stderr.write("Worst arms: %s\n" % debug_str)
        
        return err_1, err_2, err_3
        
    def explore_and_fit(self, budget=5000):
        num_sample = 50
        
        save_name = self.cache_dir + ("/beta_gp_muscale_explore_%s.pkl" % self.explore_strategy) 
        perf = []        
        
        self.fitter.warm_start_counts(self.counts0, self.counts1)
#         self.num_obs += self.counts0.sum() + self.counts1.sum()
        print ("Total count after sampling: ", self.counts0.sum() + self.counts1.sum())
        self.fit()
        
        sys.stderr.write("Explored %d examples \n" % self.num_obs)
        max_err, mean_err, hard_err = self.eval()
        sys.stderr.write("%0.4f %0.4f %0.4f\n" % (max_err, mean_err, hard_err))                
        perf.append((self.num_obs, max_err, mean_err, hard_err))
        
        while self.num_obs < budget:
            self.explore(num_sample)
            self.num_obs += num_sample
            self.fit()

            sys.stderr.write("Explored %d examples\n" % self.num_obs)
            max_err, mean_err, hard_err = self.eval()
            sys.stderr.write("%0.4f %0.4f %0.4f\n" % (max_err, mean_err, hard_err))        
            perf.append((self.num_obs, max_err, mean_err, hard_err))
            
            with open(save_name, "wb") as f:
                pickle.dump(perf, f)
