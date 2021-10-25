import math
import torch
from torch import distributions as distrs
import numpy as np
import sys
import tqdm
import pickle
import torch.nn.functional as F
import time

from . import dataset, data_fitter
from .utils import config, misc

if config.xla:
    import torch_xla.core.xla_model as xm
        
class SimpleExplorer():
    def __init__(self, dataset: dataset.Dataset, fitter: data_fitter.Fitter, cache_dir: str, 
                 device, explore_strategy='variance', sample_type="correctedwep", seed=0):
        super(SimpleExplorer, self).__init__()
        self.dataset = dataset
        self.fitter = fitter
        self.cache_dir = cache_dir
        self.dev = device
        self.explore_strategy = explore_strategy
        
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.sample_type = sample_type
        print ("Sampler type: %s" % self.sample_type)
        
        num_arms = len(dataset.arms)

        # initialize alpha, beta params for all the arms
        self.counts0, self.counts1 = torch.ones([num_arms], device=self.dev)*0.0, torch.ones([num_arms], device=self.dev)*0.0
        self.num_arms = num_arms
        self.init_state_dict = fitter.kernel_model.state_dict()
        self.num_obs = 0
    
        self.lmbda = 0.1
        self.alpha, self.beta = torch.zeros([num_arms], device=self.dev), torch.zeros([num_arms], device=self.dev)
        self.arms_tensor = torch.from_numpy(self.dataset.arms).type(torch.float32).to(self.dev)
        
        self.fitter.warm_start_counts(self.counts0, self.counts1)
        self._set_initial_counts(self.counts0, self.counts1)
            
    def _set_initial_counts(self, counts0, counts1):
        num_arms = len(self.dataset.arms)
        self.counts0, self.counts1 = counts0, counts1
        self.prior_acc = (self.counts1.sum()/(self.counts0.sum() + self.counts1.sum())).cpu().numpy()
        self.explored = np.zeros(self.dataset.num_arms)
        for ai in range(num_arms):
            if (self.counts0[ai] + self.counts1[ai]) > 0:
                self.explored[ai] = 1
    
    def unique_name(self):
        """
        Returns a string that is a unique signature of its parameters
        """
        ustr = "exp=%s" % self.explore_strategy
        if self.sample_type != "correctedwep":
            ustr += "_st=%s" % self.sample_type
        return ustr
    
    def mean_variance(self):
        mean = self.alpha/(self.beta+self.alpha)
        var = mean*(1-mean)/(self.alpha+self.beta+1)
        return mean, var
    
    def fit(self):
        self.alpha = self.counts1 + self.prior_acc*self.lmbda
        self.beta = self.counts0 + (1-self.prior_acc)*self.lmbda
        
    def explore(self, num_explore):
        sample_type = self.sample_type
        num_sample = 1
        with torch.no_grad():
            mean, variance = self.mean_variance()
            if self.explore_strategy == 'variance':
                arm_idxs = torch.argsort(-variance)
                arm_idxs = arm_idxs.cpu().numpy()
            elif self.explore_strategy == 'random':
                arm_idxs = np.arange(self.num_arms)
                np.random.shuffle(arm_idxs) 
            elif self.explore_strategy == 'svariance2':
                mask = 1 - self.explored
                av = mask*variance.cpu().numpy()
                # to add instability on arms with enough count
                av -= (1 - mask)*np.random.uniform(0, 10, size=len(av))
                arm_idxs = np.argsort(-av)
            
            x, y = [], []
            obs_arm_idxs = []
            for arm_idx in tqdm.tqdm(arm_idxs, desc="sampling..."):
                status = self.fitter.sample_arm(arm_idx, num_sample=num_sample, counts0=self.counts0, counts1=self.counts1, sample_type=sample_type)
                if status:
                    obs_arm_idxs += [arm_idx]*num_sample
                    self.explored[arm_idx] = 1.
                if len(obs_arm_idxs) >= num_explore:
                    break
            
    def eval(self):
        mean, variance = self.mean_variance()
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
        uname = self.unique_name()
        save_name = self.cache_dir + ("/simple_" + uname) 
        save_name = save_name + ("_seed=%d.pkl" % self.seed)
            
        perfs = []
        self.fit()
        
        sys.stderr.write("Explored %d examples \n" % self.num_obs)
        metric_vals = self.eval()
        sys.stderr.write("%0.4f %0.4f\n" % (metric_vals[0], metric_vals[1]))
        perfs.append((self.num_obs, metric_vals))
        
        while self.num_obs < budget:
            self.explore(num_sample)
            self.num_obs += num_sample
            self.fit()
                
            metric_vals = self.eval()
            perfs.append((self.num_obs, metric_vals))
            
            with open(save_name, "wb") as f:
                pickle.dump(perfs, f)

def estimation_ablation(args, kwargs):
    """
    Written for the purpose of ablation study of estimation and isolate the exploration
    TODO: when sampling we are keeping track of and only sampling from available indices, this is somewhat unexpected. Although, it does not change anything. It just exhausts the pool of available indices quickly.
    """    
    errs = {}
    seed = kwargs['seed']
    _dummy = SimpleExplorer(*args, **kwargs)
    uname = _dummy.unique_name()
    save_name = _dummy.cache_dir + ("/simple_" + uname) 
    save_name = save_name + ("_mega_seed=%d_estimation_ablation.pkl" % seed)
    for num_sample in np.arange(0, 5000, 100): #[500, 1500, 3000]: 
        np.random.seed(seed)
        exp = SimpleExplorer(*args, **kwargs)
        status = exp.fitter.sample(num_sample, exp.counts0, exp.counts1, exp.explored, sample_type=exp.sample_type, replace=True)

        print ("Counts: ", exp.counts0.sum(), exp.counts1.sum())
        exp.fit()
        err, err2, err3, err4, all_err = exp.eval()
        print ("NS: %d Seed: %d err: %f %f %f %f" % (num_sample, seed, err, err2, err3, err4))
        errs[num_sample] = (err, err2, err3, err4, all_err)
        with open(save_name, "wb") as f:
            pickle.dump(errs, f)
        
    with open(save_name, "wb") as f:
        pickle.dump(errs, f)
    return errs
