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

EMB_SIZE = 20
    
def mu_transform(x):
    return torch.sigmoid(6*x-3)

def rev_mu_transform(y):
    return (-torch.log(1./y - 1) + 3)/6

# def mu_transform(x):
#     return torch.sigmoid(x)

# def rev_mu_transform(y):
#     return -torch.log(1./y - 1)
    
class BernoulliLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    """
    These are the only two functions that are needed for ELBO computation
    """
    def forward(self, function_samples, **kwargs):
        # function_samples would be of shape 20(num samples) x batch_size 
        mu = mu_transform(function_samples)
        count = kwargs["count"]
        def fn(observations):
            alpha, beta = observations, count - observations
            log_prob = alpha*torch.log(mu + 1e-10) + beta*torch.log(1 - mu + 1e-10)
            return log_prob 
        return fn
    
    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        log_prob_lambda = lambda function_samples: self.forward(function_samples, **kwargs)(observations)
        prior_acc = kwargs["prior_acc"]
        def prior_log_prob_fn(function_samples):
            mu = mu_transform(function_samples)
            log_prob = 0.1*(prior_acc*torch.log(mu + 1e-10) + (1-prior_acc)*torch.log(1 - mu + 1e-10))
            return log_prob

        log_prob = self.quadrature(log_prob_lambda, function_dist)
        prior_log_prob = self.quadrature(prior_log_prob_fn, function_dist)
        return log_prob + prior_log_prob


class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x, embedder, prior_acc):
        inducing_points = torch.randn([50, train_x.size(1)])
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        
        variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            )
        super().__init__(variational_strategy)
        
        prior = gpytorch.priors.UniformPrior(0, 1)
        ls_prior = gpytorch.priors.UniformPrior(torch.tensor(1e-3), torch.tensor(1e-2))
        ls_cstr = gpytorch.constraints.Interval(1-1e-2, 1.)
        # serves as the prior for the inducing points.
        self.mean_module = gpytorch.means.ConstantMean(prior=prior)
        self.kernel = gpytorch.kernels.RBFKernel(ard_num_dims=EMB_SIZE)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.kernel
        )
        self.embedder = embedder

    def forward(self, x):
        embeds = self.embedder(x)
        mean_x = self.mean_module(embeds)
        covar_x = self.covar_module(embeds)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
    
    
class BernGPExplorer(torch.nn.Module):
    def __init__(self, dataset: dataset.Dataset, fitter: data_fitter.Fitter, cache_dir: str, 
                 device, explore_strategy='variance', fit_strategy='gp', err_ns=[.03, .05, .1], seed=0, gp_emb_size=EMB_SIZE, num_updates=50, num_ws_updates=5000, lr=1e-3):
        super(BernGPExplorer, self).__init__()
        self.err_ns = err_ns
        self.dataset = dataset
        self.fitter = fitter
        self.cache_dir = cache_dir
        self.dev = device
        self.explore_strategy = explore_strategy
        self.fit_strategy = fit_strategy
        self.seed = seed
        self.num_fit_updates = num_updates
        self.num_ws_updates = num_ws_updates
        np.random.seed(seed)
        torch.manual_seed(seed)
        global EMB_SIZE
        EMB_SIZE = gp_emb_size
        
        num_arms = len(dataset.arms)

        # initialize alpha, beta params for all the arms
        self.counts0, self.counts1 = torch.ones([num_arms], device=self.dev)*0.0, torch.ones([num_arms], device=self.dev)*0.0
        self.num_arms = num_arms
        self.init_state_dict = fitter.kernel_model.state_dict()
        self.num_obs = 0
    
        linear_layer1 = torch.nn.Linear(self.fitter.kernel_model.emb_size, gp_emb_size)
        self.embedder = torch.nn.Sequential(
            self.fitter.kernel_model.embedding_model,
            torch.nn.ReLU(),
            linear_layer1,
        ).to(self.dev)
        self.arms_tensor = torch.from_numpy(self.dataset.arms).type(torch.float32).to(self.dev)
        
        self.fitter.warm_start_counts(self.counts0, self.counts1)
        self.prior_acc = (self.counts1.sum()/(self.counts0.sum() + self.counts1.sum())).numpy()        
        self.explored = np.zeros(self.dataset.num_arms)
        for ai in range(num_arms):
            if (self.counts0[ai] + self.counts1[ai]) > 0:
                self.explored[ai] = 1

        # Initialize model and likelihood
        self.gp_model = GPClassificationModel(self.arms_tensor, self.embedder, self.prior_acc).to(self.dev)
        self.gp_likelihood = BernoulliLikelihood()

        all_vars = list(self.gp_model.named_parameters()) + list(self.embedder.named_parameters())
        length_params = []
        all_but_length_params = []
        l_names = ["kernel.raw_lengthscale", "covar_module.raw_outputscale"]
        gp_params, embedder_params = [], []
        for name, p in self.gp_model.named_parameters():
            if name.startswith('embedder'):
                embedder_params.append(p)
            else:
                gp_params.append(p)

        self.length_optimizer = torch.optim.Adam(list(zip(*all_vars))[1], lr=1e-3, weight_decay=0)
        self.optimizer1 = torch.optim.Adam(gp_params + embedder_params, lr=lr, weight_decay=0)
        self.optimizer2 = torch.optim.Adam(gp_params, lr=lr, weight_decay=0)

        # for 1D integrals 
        self.quadrature = gpytorch.utils.quadrature.GaussHermiteQuadrature1D()
    
    def _init_params(self):
        for param in self.embedder.parameters():
            torch.nn.init.normal_(param, 0, 1e-3)
        
    def mean_variance(self, debug=False):
        self.gp_model.eval()
        self.embedder.eval()
        self.gp_likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            count = self.counts0 + self.counts1
            z = self.gp_model(self.arms_tensor)
            latent_mean, latent_variance = z.mean, z.variance
            normal_dist = distrs.Normal(latent_mean, latent_variance)

            def mean_fn(sample):
                mu = mu_transform(sample)
                return mu
            mean = self.quadrature(mean_fn, normal_dist)

            def variance_fn(sample):
                # sample of shape: [20 x num_arms]
                diff = mu_transform(sample) - torch.unsqueeze(mean, dim=0)
                return diff**2
            variance = self.quadrature(variance_fn, normal_dist)
            def variance_fn2(sample):
                mean = torch.mean(sample, dim=0)
                # sample of shape: [20 x num_arms]
                diff = sample - torch.unsqueeze(mean, dim=0)
                return diff**2
            
            with open(self.cache_dir + "/test_bern_gp.pkl", "wb") as f:
                pickle.dump((mean.numpy(), self.counts0.numpy(), self.counts1.numpy()), f)
            if debug:
                return mean, variance, latent_mean, latent_variance
            else:
                return mean, variance
    
    def fit(self, num_updates=200): 
        self.gp_model.train()
        self.gp_likelihood.train()
        
        for _ in range(2):
            obs_idxs = np.where(self.explored == 1)[0]
            if _==1:
                all_idxs = np.array([idx for idx, acc in enumerate(self.fitter.arm_index_to_acc) if acc is not np.nan])
                obs_idxs = np.union1d(obs_idxs, all_idxs)
            count = (self.counts0 + self.counts1)[obs_idxs]
            obs_y = self.counts1[obs_idxs]

            mll = gpytorch.mlls.VariationalELBO(self.gp_likelihood, self.gp_model, obs_y.numel(), beta=1, combine_terms=True)

            if _ == 0:
                optimizer = self.optimizer1
            else:
                optimizer = self.optimizer2
            for i in range(num_updates):
                optimizer.zero_grad()
                latent_mean = self.gp_model(self.arms_tensor[obs_idxs]).mean
                output = self.gp_model(self.arms_tensor[obs_idxs])
                loss = -mll(output, obs_y, count=count, prior_acc=torch.tensor(self.prior_acc))
                loss.backward()
                optimizer.step()
        
        # debug stuff
        mean, variance, a, b = self.mean_variance(debug=True)
        emp_mean = obs_y/count
        errs = np.abs((emp_mean - mean[obs_idxs]).numpy())
        _idx, idx = np.argmax(errs), obs_idxs[np.argmax(errs)]
        
        _c0, _c1 = self.counts0.cpu().numpy(), self.counts1.cpu().numpy()
        bad_arms = np.argsort(self.fitter.arm_index_to_acc)
        debug_str = str(["%d %0.4f %0.2f %0.4f" % (ai, mean[ai], (_c1[ai]/(_c1[ai]+_c0[ai]) if (_c0[ai]+_c1[ai])>0 else np.nan), self.fitter.arm_index_to_acc[ai]) for ai in bad_arms[:50]])
        sys.stderr.write("Worst arms: %s\n" % debug_str)
        print ("Median, mean error:", np.median(errs), np.mean(errs))
        print ("")

    def explore(self, num_explore):
        sample_type = "correctedwep"
        with torch.no_grad():
            mean, variance, a, latent_variance = self.mean_variance(debug=True)
            if self.explore_strategy == "wvariance":
                self.fitter.sample_from_arms(variance, num_explore, self.counts0, self.counts1, sample_type=sample_type)
            else:
                num_sample = 1
                if self.explore_strategy == 'variance':
                    arm_idxs = torch.argsort(-variance)
                    arm_idxs = arm_idxs.cpu().numpy()
                elif self.explore_strategy == 'svariance':
                    av = self.fitter.arm_availability(sample_type)
                    av *= variance.cpu().numpy()
                    arm_idxs = np.argsort(-av)
                elif self.explore_strategy == 'svariance2':
                    mask = 1 - self.explored
                    av = mask*variance.cpu().numpy()
                    # to add instability on arms with enough count
                    av -= (1 - mask)*np.random.uniform(0, 10, size=len(av))
                    arm_idxs = np.argsort(-av)
                elif self.explore_strategy == 'random':
                    arm_idxs = np.arange(self.num_arms)
                    np.random.shuffle(arm_idxs) 
            
                x, y = [], []
                obs_arm_idxs = []
                for arm_idx in tqdm.tqdm(arm_idxs, desc="sampling..."):
                    # correctedwep
                    status = self.fitter.sample_arm(arm_idx, num_sample=num_sample, counts0=self.counts0, counts1=self.counts1, sample_type=sample_type)
                    if status:
                        obs_arm_idxs += [arm_idx]*num_sample
                        self.explored[arm_idx] = 1.
                    if len(obs_arm_idxs) >= num_explore:
                        break

                _c0, _c1 = self.counts0.cpu().numpy(), self.counts1.cpu().numpy()
                sys.stderr.write("Err of arms being explored: %s\n" % str([(ai, _c0[ai], _c1[ai], a[ai], variance[ai]) for ai in obs_arm_idxs]))
                _s1 =  np.sort(a.detach().numpy())
                sys.stderr.write("%s %s\n" % (_s1[-10:], _s1[:10]))
            
    def eval(self):
        mean, variance, latent_mean, latent_variance = self.mean_variance(debug=True)
        if self.fit_strategy == "gp":
            mu_hat = mean.detach().cpu().numpy()
        else:
            mu_hat = self.simple_fit()
        err_1, all_err = self.fitter.evaluate_mean_estimates(mu_hat, self.err_ns[0], debug=True)
        err_2, err_3 = [self.fitter.evaluate_mean_estimates(mu_hat, _)[0] for _ in self.err_ns[1:]]
        
        all_err = np.array(all_err)
        bad_arms = np.argsort(-all_err)
        _c0, _c1 = self.counts0.cpu().numpy(), self.counts1.cpu().numpy()
        
        debug_obj = [(ai, all_err[ai], self.fitter.arm_index_to_acc[ai], _c0[ai], _c1[ai]) for ai in bad_arms]
        fname = "%s/debug_bern_gp_%s.pkl" % (self.cache_dir, self.explore_strategy)

        with open(fname, "wb") as f:
            pickle.dump(debug_obj, f)
        
#         debug_str = str(["%d %0.4f %0.4f %0.4f %0.4f %0.3f %0.3f" % (ai, mean[ai], latent_variance[ai], self.fitter.arm_index_to_acc[ai], all_err[ai], _c0[ai], _c1[ai]) for ai in bad_arms[:50]])
#         sys.stderr.write("Worst arms: %s\n" % debug_str)
        
        return err_1, err_2, err_3
       
    def simple_fit(self):
        counts0 = self.counts0.numpy().copy()
        counts1 = self.counts1.numpy().copy()
        counts0 += (1 - self.prior_acc)*0.1
        counts1 += self.prior_acc*0.1
        
        return counts1/(counts0 + counts1)
        
    def explore_and_fit(self, budget=5000):
        num_sample = 12
        
        print ("Number of unique arms: ", len(self.fitter.dataset.arm_to_idxs))
        if self.fit_strategy == "gp":
            save_name = self.cache_dir + ("/bern_gp_explore_%s_seed=%d_2.pkl" % (self.explore_strategy, self.seed)) 
        else:
            save_name = "%s/bern_gp_explore_ft=%s_et=%s_seed=%d_2.pkl" % (self.cache_dir, self.fit_strategy, self.explore_strategy, self.seed)

        perf = []
        print ("Total count after sampling: ", self.counts0.sum() + self.counts1.sum())
        self.fit(1000)
        
        sys.stderr.write("Explored %d examples \n" % self.num_obs)
        max_err, mean_err, hard_err = self.eval()
        sys.stderr.write("%0.4f %0.4f %0.4f\n" % (max_err, mean_err, hard_err))                
        perf.append((self.num_obs, max_err, mean_err, hard_err))
        
        while self.num_obs < budget:
            self.explore(num_sample)
            self.num_obs += num_sample
            self.fit(100)
                
            sys.stderr.write("Explored %d examples\n" % self.num_obs)
            max_err, mean_err, hard_err = self.eval()
            sys.stderr.write("%0.4f %0.4f %0.4f\n" % (max_err, mean_err, hard_err))        
            perf.append((self.num_obs, max_err, mean_err, hard_err))
            
            with open(save_name, "wb") as f:
                pickle.dump(perf, f)
