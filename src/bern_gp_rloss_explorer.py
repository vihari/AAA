import math
import torch
from torch import distributions as distrs
import numpy as np
import sys
import tqdm
import pickle
import torch.nn.functional as F
import time

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy

from . import dataset, data_fitter
from .utils import config, misc
from .likelihoods import bern_gp_likelihood

if config.xla:
    import torch_xla.core.xla_model as xm

EMB_SIZE = 20

"""
This is a clone of bern_gp_explorer.py for experimentation on the region loss.
"""

S = None
def rev_mu_transform(y):
    return (-torch.log(1./y - 1) + 3)/6

def get_smoothing_matrix(kernel_widths, num_obs):
    """
    returns number of new obs x num_obs
    """
    s_matrix = []
    for width in kernel_widths:
        ln = num_obs - width + 1
        k = np.zeros([ln, num_obs])
        for _ in range(ln):
            k[_, _:_+width] = 1.
        s_matrix.append(k)
    s_matrix = torch.tensor(np.concatenate(s_matrix, axis=0)).type(torch.float32)
    assert len(s_matrix) == (len(kernel_widths)*num_obs - sum(kernel_widths) + len(kernel_widths))
    # randomize
    rand_idx = np.random.permutation(num_obs)
    s_matrix = s_matrix[:, rand_idx]
    return s_matrix

def get_region(seed, counts, max_width, p):
    ni = seed
    num_obs = len(counts)
    _sm = np.zeros(num_obs)
    
    p /= p.sum()
    idxs = [ni] + np.random.choice(num_obs, max_width-1, p=p).tolist()
    support = 0
    for idx in idxs:
        support += counts[idx]
        _sm[idx] = 1. # np.random.uniform(0, 1)
    _sm /= _sm.sum()
    return _sm

def get_random_smoothing_matrix(counts, width=3):
    """
    returns number of new obs x num_obs
    """
    st = time.time()
    num_obs = len(counts)
    s_matrix = []
    lns = []
    if type(width) != list:
        width = [(width, 2*width)]
    for ni in range(num_obs):
        for _w, nrs in width:
            p = np.ones(len(counts))
            p[ni] = 0
            if counts[ni] >= _w:
                region = get_region(ni, counts, 1, p)
                s_matrix.append(region)
            for nr in range(nrs):
                region = get_region(ni, counts, _w, p)
                s_matrix.append(region)
                lns.append(len(np.nonzero(region)[0]))
            for nr in range(nrs):
                region = get_region(ni, counts, _w, p)
                s_matrix.append(region)
                lns.append(len(np.nonzero(region)[0]))
    S = np.stack(s_matrix, axis=0).astype(np.float32)
    assert np.alltrue(S.sum(axis=-1)>0)   
    return torch.from_numpy(S)

def get_random_smoothing_matrix2(kernel_widths, counts, num_regions=5):
    """
    Smoothing matrix carefully made to avoid high condition number of weighted smoothing matrix 
    """
    s_matrix = []
    regions = []
    scount_idxs = np.argsort(counts.numpy())
    region_start, this_region = None, None
    idx_to_region_mapping = {}
    for _ii, idx in enumerate(scount_idxs):
        if region_start is None:
            assert _ii==0
            region_start = counts[idx]
            this_region = []
        if counts[idx]/region_start > 2:
            regions.append(this_region)
            region_start = counts[idx]
            this_region = []
        this_region.append(idx)
        idx_to_region_mapping[idx] = len(regions)
    regions.append(this_region)
    
    num_obs = len(counts)
    for ni in range(num_obs):
        rid = idx_to_region_mapping[ni]
        for width in kernel_widths:
            for wi in range(num_regions):
                s_row = np.zeros([num_obs])
                _w = min(len(regions[rid]), width-1)
                idxs = [ni] + np.random.choice(regions[rid], _w, replace=False).tolist()
                for idx in idxs:
                    s_row[idx] = 1./(_w+1)
                s_matrix.append(s_row)
    S = np.stack(s_matrix, axis=0).astype(np.float32)
    
    assert np.alltrue(S.sum(axis=-1)>0)    
    return torch.from_numpy(S)

def get_support_region(counts, threshs):
    """
    :param num_per_example: The number of regions an example should fall in.
    """
    S = np.zeros([len(threshs)*len(counts), len(counts)], dtype=np.float32)
    nums = np.zeros([len(threshs)*len(counts)])
    for ci in range(len(counts)):
        start_idx = len(threshs)*ci
        for k, thresh in enumerate(threshs):
            S[start_idx + k, ci] = 1.
            support = counts[ci]
            num = 1
            while support <= thresh:
                rand_idx = np.random.choice(len(counts))
                if rand_idx == ci:
                    continue
                S[start_idx + k, rand_idx] = 1.
                support += counts[rand_idx]
                num += 1
            nums[start_idx + k] = num
            
    S /= np.expand_dims(nums, axis=-1)
    assert np.alltrue(S.sum(axis=-1)>0)
    return torch.from_numpy(S)

class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x, embedder, prior_acc):
        dev = train_x.device
        inducing_points = torch.randn([50, train_x.size(1)]).to(dev)
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
        self.embedder = embedder.to(dev)

    def forward(self, x):
        embeds = self.embedder(x)
        mean_x = self.mean_module(embeds)
        covar_x = self.covar_module(embeds)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
    
    
class BernGPExplorer(torch.nn.Module):
    def __init__(self, dataset: dataset.Dataset, fitter: data_fitter.Fitter, cache_dir: str, 
                 device, explore_strategy='variance', fit_strategy='gp', err_ns=[.03, .05, .1], seed=0, gp_emb_size=EMB_SIZE, num_updates=50, num_ws_updates=5000, lr=1e-3, width=3, sample_type="correctedwep"):
        super(BernGPExplorer, self).__init__()
        device = torch.device("cpu")
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
        self.width = width
        self.sample_type = sample_type
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
        self._set_initial_counts(self.counts0, self.counts1)
        
        # Initialize model and likelihood
        self.gp_model = GPClassificationModel(self.arms_tensor, self.embedder, self.prior_acc).to(self.dev)
        self.gp_likelihood = bern_gp_likelihood.BernLikelihood(self.width).to(self.dev)

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
        self.quadrature = gpytorch.utils.quadrature.GaussHermiteQuadrature1D().to(self.dev)
    
    def _set_initial_counts(self, counts0, counts1):
        num_arms = len(self.dataset.arms)
        self.counts0, self.counts1 = counts0, counts1
        self.prior_acc = (self.counts1.sum()/(self.counts0.sum() + self.counts1.sum())).cpu().numpy()
        self.explored = np.zeros(self.dataset.num_arms)
        for ai in range(num_arms):
            if (self.counts0[ai] + self.counts1[ai]) > 0:
                self.explored[ai] = 1
    
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
                mu = misc.mu_transform(sample)
                return mu
            mean = self.quadrature(mean_fn, normal_dist)

            def variance_fn(sample):
                # sample of shape: [20 x num_arms]
                diff = misc.mu_transform(sample) - torch.unsqueeze(mean, dim=0)
                return diff**2
            variance = self.quadrature(variance_fn, normal_dist)
            def variance_fn2(sample):
                mean = torch.mean(sample, dim=0)
                # sample of shape: [20 x num_arms]
                diff = sample - torch.unsqueeze(mean, dim=0)
                return diff**2
            
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
            count = (self.counts0 + self.counts1)[obs_idxs].to(self.dev)
            obs_y = self.counts1[obs_idxs].to(self.dev)

            mll = gpytorch.mlls.VariationalELBO(self.gp_likelihood, self.gp_model, obs_y.numel(), beta=1, combine_terms=True)

            global S
            S = get_random_smoothing_matrix(counts=count[torch.where(count>0)[0]].cpu().numpy(), width=self.width)
            S /= torch.unsqueeze(S.sum(dim=-1), dim=-1)
            S = S.to(self.dev)
            
            print ("Fitting on:", len(obs_idxs))
            if _ == 0:
                optimizer = self.optimizer1
                smooth = False
            else:
                optimizer = self.optimizer2
                smooth = True
            print ("Min count: ", torch.min(count))
            for i in tqdm.tqdm(range(num_updates)):
                optimizer.zero_grad()
                latent_mean = self.gp_model(self.arms_tensor[obs_idxs]).mean
                output = self.gp_model(self.arms_tensor[obs_idxs])
                loss = -mll(output, obs_y, count=count, smooth=smooth, prior_acc=torch.tensor(self.prior_acc), S=S)
                loss.backward()
                optimizer.step()
        
        # debug stuff
        mean, variance, a, b = self.mean_variance(debug=True)
        mean, variance = mean.cpu(), variance.cpu()
        emp_mean = obs_y/count
        errs = np.abs((emp_mean - mean[obs_idxs]).numpy())
        _idx, idx = np.argmax(errs), obs_idxs[np.argmax(errs)]
        
        _c0, _c1 = self.counts0.cpu().numpy(), self.counts1.cpu().numpy()
        ds = []
        errs2 = np.abs((torch.tensor(self.fitter.arm_index_to_acc) - mean).cpu().numpy())
        bad_arms = np.argsort(-errs2)
        for ai in bad_arms:
            if self.fitter.arm_index_to_acc[ai] is not np.nan:
                ds.append("%d %0.4f %0.2f %0.4f" % (ai, mean[ai], (_c1[ai]/(_c1[ai]+_c0[ai]) if (_c0[ai]+_c1[ai])>0 else np.nan), self.fitter.arm_index_to_acc[ai]))
            if len(ds) > 50:
                break
                           
        debug_str = str(ds)
        sys.stderr.write("Worst arms: %s\n" % debug_str)
        print ("Median, mean error:", np.median(errs), np.mean(errs))
        print ("")

    def explore(self, num_explore):
        sample_type = self.sample_type
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
                _s1 =  np.sort(a.detach().cpu().numpy())
                sys.stderr.write("%s %s\n" % (_s1[-10:], _s1[:10]))
            
    def eval(self):
        mean, variance, latent_mean, latent_variance = self.mean_variance(debug=True)
        mu_hat = mean.detach().cpu().numpy()

        # all_err is sorted from lowest arm acc to the left and highest to right
        mse = self.fitter.evaluate_mean_estimates(mu_hat, 0)[0]
        micro_mse = self.fitter.evaluate_mean_micromse(mu_hat)
        worst_err, all_err = self.fitter.evaluate_mean_worstacc(mu_hat)
        worst_err2 = self.fitter.evaluate_mean_worstfreq(mu_hat)
        return mse, micro_mse, worst_err, worst_err2, all_err
    
    def explore_and_fit(self, budget=5000):
        num_sample = 12
        
        print ("Number of unique arms: ", len(self.fitter.dataset.arm_to_idxs))
        if self.fit_strategy == "gp":
            save_name = self.cache_dir + ("/bern_gp_explore_%s_width=%s_seed=%d.pkl" % (self.explore_strategy, str(self.width), self.seed)) 
        else:
            save_name = "%s/bern_gp_explore_et=%s_width=%s_seed=%d.pkl" % (self.cache_dir, self.fit_strategy, self.explore_strategy, str(self.width), self.seed)

        perf = []
        print ("Total count after sampling: ", self.counts0.sum() + self.counts1.sum())
        self.fit(1000)
        
        sys.stderr.write("Explored %d examples \n" % self.num_obs)
        errs = self.eval()
        sys.stderr.write("%0.4f %0.4f %0.4f\n" % (errs[0], errs[1], errs[2]))                
        perf.append((self.num_obs, errs))
        
        while self.num_obs < budget:
            self.explore(num_sample)
            self.num_obs += num_sample
            self.fit(100)
                
            sys.stderr.write("Explored %d examples\n" % self.num_obs)
            errs = self.eval()
            sys.stderr.write("%0.4f %0.4f %0.4f\n" % (errs[0], errs[1], errs[2]))               
            perf.append((self.num_obs, errs))
            
            with open(save_name, "wb") as f:
                pickle.dump(perf, f)

def estimation_ablation(args, kwargs):
    """
    Written for the purpose of ablation study of estimation and isolate the exploration
    """
    errs = {}
    seed = kwargs['seed']
    save_name = "%s/bern_gp_explorer_width=%d_seed=%d_estimation_ablation.pkl" % (args[1].cache_dir, kwargs['width'], seed)
    for num_sample in [500, 1500, 3000]: 
        _errs = []
        np.random.seed(seed)

        exp = BernGPExplorer(*args, **kwargs)
        status = exp.fitter.sample(num_sample, exp.counts0, exp.counts1, exp.explored)

        print (exp.counts0.sum(), exp.counts1.sum())
        exp.fit(1000)
        mvals = exp.eval()
        print ("NS: %d Seed: %d err: %s" % (num_sample, seed, mvals[:-1]))
        
        errs[num_sample] = mvals
        with open(save_name, "wb") as f:
            pickle.dump(errs, f)        

    with open(save_name, "wb") as f:
        pickle.dump(errs, f)
    return errs