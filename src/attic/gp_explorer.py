import math
import torch
from torch import distributions as distrs
import numpy as np
import sys
import tqdm
import pickle
import torch.nn.functional as F

import gpytorch
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy

from . import dataset, data_fitter
from .utils import config

if config.xla:
    import torch_xla.core.xla_model as xm

EMB_SIZE = 50

    
class GPRegressionModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, embedder):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        prior = gpytorch.priors.NormalPrior(loc=0.5, scale=1.)
        self.mean_module = gpytorch.means.ConstantMean(prior=prior)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(), ard_num_dims=EMB_SIZE)
        self.embedder = embedder

    def forward(self, x):
        embeds = self.embedder(x)
        mean_x = self.mean_module(embeds)
        covar_x = self.covar_module(embeds)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
    
    
class GPExplorer(torch.nn.Module):
    def __init__(self, dataset: dataset.Dataset, fitter: data_fitter.Fitter, 
                 cache_dir: str, device, explore_strategy='variance', err_ns=[.03, .05, .1]):
        super(GPExplorer, self).__init__()
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

        # likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        self.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
        # for 1D integrals 
        self.quadrature = gpytorch.utils.quadrature.GaussHermiteQuadrature1D()
    
    def _init_params(self):
        for param in self.embedder.parameters():
            torch.nn.init.normal_(param, 0, 1e-3)
        self.fitter.kernel_model.load_state_dict(self.init_state_dict)
    
    def mean_variance(self):
        self.gp_model.eval()
        self.embedder.eval()
        self.gp_likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            z = self.gp_model(self.arms_tensor)
            latent_mean, latent_variance = z.mean, z.variance
            _m = latent_mean.numpy()
            
            with open(self.cache_dir + "/test_gp.pkl", "wb") as f:
                pickle.dump((_m, self.counts0.numpy(), self.counts1.numpy()), f)
            return latent_mean, latent_variance
    
    def fit(self):         
        obs_idxs = torch.where(self.counts0 + self.counts1 > 0)[0]
        count = (self.counts0 + self.counts1)[obs_idxs]
        obs_y = self.counts1[obs_idxs]/count
        # Initialize model and likelihood
        self.gp_model = GPRegressionModel(self.arms_tensor[obs_idxs], obs_y, self.gp_likelihood, self.embedder).to(self.dev)

        self.gp_model.train()
        self.gp_likelihood.train()
        
        self.optimizer = torch.optim.Adam(list(self.gp_model.parameters()) + list(self.embedder.parameters()), lr=1e-3, weight_decay=0)
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model)
#         mll = gpytorch.mlls.VariationalELBO(self.gp_likelihood, self.gp_model, obs_y.numel())
        
        training_iterations = 200
        for i in range(training_iterations):
            self.optimizer.zero_grad()
            arm_embeddings = self.embedder(self.arms_tensor[obs_idxs])
            output = self.gp_model(self.arms_tensor[obs_idxs])
            loss = -mll(output, obs_y)
            loss.backward()
            self.optimizer.step()
            
        mean, variance = self.mean_variance()
        emp_mean = self.counts1[obs_idxs]/count
        errs = np.abs((emp_mean - mean[obs_idxs]).numpy())
        _idx, idx = np.argmax(errs), obs_idxs[np.argmax(errs)]
        print ("Worst error: %0.4f counts: %f %f" % (errs[_idx], self.counts0[idx], self.counts1[idx]))
        print ("Median, mean error:", np.median(errs), np.mean(errs))
        print ("")

    def explore(self, num_explore):        
        with torch.no_grad():
            if hasattr(self, "gp_model"): 
                mean, variance = self.mean_variance()
            if self.explore_strategy == 'variance':
                arm_idxs = torch.argsort(-variance)
                arm_idxs = arm_idxs.cpu().numpy()
            elif self.explore_strategy == 'random':
                arm_idxs = np.arange(self.num_arms)
                np.random.shuffle(arm_idxs) 
            
            x, y = [], []
            obs_arm_idxs = []
            num_sample = 5
            for arm_idx in arm_idxs:
                status = self.fitter.sample_arm(arm_idx, num_sample=num_sample, counts0=self.counts0, counts1=self.counts1)
                if status:
                    obs_arm_idxs += [arm_idx]*num_sample
                if len(obs_arm_idxs) >= num_explore:
                    break

            _c0, _c1 = self.counts0.cpu().numpy(), self.counts1.cpu().numpy()
            if hasattr(self, "gp_model"): 
                sys.stderr.write("Err of arms being exploreds: %s\n" % str([(ai, _c0[ai], _c1[ai], mean[ai], variance[ai]) for ai in obs_arm_idxs]))

    def eval(self):
        mean, variance = self.mean_variance()
        mu_hat = mean.detach().cpu().numpy()
        err_1, all_err = self.fitter.evaluate_mean_estimates(mu_hat, self.err_ns[0], debug=True)
        err_2, err_3 = [self.fitter.evaluate_mean_estimates(mu_hat, _)[0] for _ in self.err_ns[1:]]
        
        bad_arms = np.argsort(-np.array(all_err))
        _c0, _c1 = self.counts0.cpu().numpy(), self.counts1.cpu().numpy()
        debug_str = str(["%d %0.4f %0.4f %0.3f %0.3f" % (ai, all_err[ai], self.fitter.arm_index_to_acc[ai], _c0[ai], _c1[ai]) for ai in bad_arms[:50]])
        sys.stderr.write("Worst arms: %s\n" % debug_str)
        
        return err_1, err_2, err_3
        
    def explore_and_fit(self, budget=5000):
        num_sample = 50
        
        save_name = self.cache_dir + ("/gp_exact_explore_%s.pkl" % self.explore_strategy) 
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
