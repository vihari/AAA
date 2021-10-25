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

    
class BetaBinomialLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    """
    These are the only two functions that are needed for ELBO computation
    """
    def forward(self, function_samples, **kwargs):
        count, scale = kwargs["count"], kwargs["scale"]
        mus = torch.sigmoid(function_samples)
        alphas, betas = mus*scale + 1, (1-mus)*scale + 1
        return gpytorch.distributions.base_distributions.BetaBinomial(alphas, betas, count)
    
    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        log_prob_lambda = lambda function_samples: self.forward(function_samples, **kwargs).log_prob(observations)
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob

    
class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x, embedder):
        variational_distribution = CholeskyVariationalDistribution(100)
        variational_strategy = UnwhitenedVariationalStrategy(
            self, torch.randn([100, train_x.size(1)]), variational_distribution, learn_inducing_locations=True
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.embedder = embedder

    def forward(self, x):
        embeds = self.embedder(x)
        mean_x = self.mean_module(embeds)
        covar_x = self.covar_module(embeds)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
    
    
class BetaGPExplorer(torch.nn.Module):
    def __init__(self, dataset: dataset.Dataset, fitter: data_fitter.Fitter, cache_dir: str, device, explore_strategy='variance'):
        super(BetaGPExplorer, self).__init__()
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
        self._init_params()
        self.num_obs = 0
    
        linear_layer = torch.nn.Linear(20, 10)
        self.embedder = torch.nn.Sequential(
            self.fitter.kernel_model.embedding_model,
            torch.nn.ReLU(),
            linear_layer,
        ).to(self.dev)
        self.arms_tensor = torch.from_numpy(self.dataset.arms).type(torch.float32).to(self.dev)
        self.scale_covar_module = gpytorch.kernels.RBFKernel()

        # Initialize model and likelihood
        self.gp_model = GPClassificationModel(self.arms_tensor, self.embedder).to(self.dev)
        # likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        self.gp_likelihood = BetaBinomialLikelihood()

        self.optimizer = torch.optim.Adam(list(self.gp_model.parameters()) + list(self.embedder.parameters()) + list(self.scale_covar_module.parameters()), lr=1e-3)
    
        # for 1D integrals 
        self.quadrature = gpytorch.utils.quadrature.GaussHermiteQuadrature1D()
    
    def _init_params(self):
        self.fitter.kernel_model.load_state_dict(self.init_state_dict)
    
    def mean_variance(self):
        self.gp_model.eval()
        self.embedder.eval()
        self.gp_likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            z = self.gp_model(self.arms_tensor)
            latent_mean, latent_variance = z.mean, z.variance
            normal_dist = distrs.Normal(latent_mean, latent_variance)

            count = self.counts0 + self.counts1
            scale_kernel = self.scale_covar_module(self.embedder(self.arms_tensor)).evaluate()
            # get smoothed scale value
            smoothed_scale = torch.mv(scale_kernel, count)

            def mean_fn(sample):
                return torch.sigmoid(sample)
            def variance_fn(sample):
                mu = torch.sigmoid(sample)
                return mu*(1-mu)/smoothed_scale
            mean = self.quadrature(mean_fn, normal_dist)
            variance = self.quadrature(variance_fn, normal_dist)
            return mean, variance, {"scale": smoothed_scale}
    
    def fit(self): 
        self.gp_model.train()
        self.gp_likelihood.train()
        
        obs_idxs = torch.where(self.counts0 + self.counts1 > 0)[0]
        count = (self.counts0 + self.counts1)[obs_idxs]
        obs_y = self.counts1[obs_idxs]
        
        mll = gpytorch.mlls.VariationalELBO(self.gp_likelihood, self.gp_model, obs_y.numel())
        
        training_iterations = 200
        for i in range(training_iterations):
            self.optimizer.zero_grad()
            arm_embeddings = self.embedder(self.arms_tensor[obs_idxs])
            output = self.gp_model(self.arms_tensor[obs_idxs])
            scale_kernel = self.scale_covar_module(arm_embeddings.detach()).evaluate()
            scale = torch.mv(scale_kernel, count)
            loss = -mll(output, obs_y, count=count, scale=scale)
            loss.backward()
            self.optimizer.step()

    def explore(self, num_explore):
        with torch.no_grad():
            mean, variance, debug = self.mean_variance()
            if self.explore_strategy == 'variance':
                arm_idxs = torch.argsort(-variance)
                arm_idxs = arm_idxs.cpu().numpy()
            elif self.explore_strategy == 'random':
                arm_idxs = np.arange(self.num_arms)
                np.random.shuffle(arm_idxs) 
            
            x, y = [], []
            obs_arm_idxs = []
            num_sample = 5
            for arm_idx in tqdm.tqdm(arm_idxs, desc="sampling..."):
                status = self.fitter.sample_arm(arm_idx, num_sample=num_sample, counts0=self.counts0, counts1=self.counts1)
                if status:
                    obs_arm_idxs += [arm_idx]*num_sample
                if len(obs_arm_idxs) >= num_explore:
                    break

            _c0, _c1 = self.counts0.cpu().numpy(), self.counts1.cpu().numpy()
            _s = debug["scale"].detach().cpu().numpy()
            sys.stderr.write("Err of arms being exploreds: %s\n" % str([(ai, _c0[ai], _c1[ai], _s[ai]) for ai in obs_arm_idxs]))

    def eval(self):
        mean, variance, _ = self.mean_variance()
        mu_hat = mean.detach().cpu().numpy()
        err_3, all_err = self.fitter.evaluate_mean_estimates(mu_hat, 0.03, debug=True)
        err_5, err_10 = [self.fitter.evaluate_mean_estimates(mu_hat, _)[0] for _ in [0.05, .1]]
        
        bad_arms = np.argsort(-np.array(all_err))
        _c0, _c1 = self.counts0.cpu().numpy(), self.counts1.cpu().numpy()
        debug_str = str(["%d %0.4f %0.4f %0.3f %0.3f" % (ai, all_err[ai], self.fitter.arm_index_to_acc[ai], _c0[ai], _c1[ai]) for ai in bad_arms[:50]])
        sys.stderr.write("Worst arms: %s\n" % debug_str)
        
        return err_3, err_5, err_10
        
    def explore_and_fit(self, budget=5000):
        ws_num = 500
        num_sample = 50
        
        save_name = self.cache_dir + ("/beta_gp_explore_%s.pkl" % self.explore_strategy) 
        perf = []
        # draw 100 examples randomly and fit
        print ("Total count before sampling: ", self.counts0.sum() + self.counts1.sum())
        original_strategy = self.explore_strategy
        self.explore_strategy = "random"
        self.explore(ws_num)
        self.explore_strategy = original_strategy

#         status = self.fitter.sample(ws_num, self.counts0, self.counts1)
        self.num_obs += ws_num
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
