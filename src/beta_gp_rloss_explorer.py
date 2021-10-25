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
from .likelihoods import beta_gp_likelihood

if config.xla:
    import torch_xla.core.xla_model as xm

EMB_SIZE = 20

"""
Beta fit (mu-scale parameterization) with region loss
"""

S = None
    
class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x, embedder):
        dev = train_x.device
        inducing_points = torch.randn([2, 50, train_x.size(1)]).to(train_x.device)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-2), batch_shape=torch.Size([2]))
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ), num_tasks=2
        )
        super().__init__(variational_strategy)
        
        prior = gpytorch.priors.NormalPrior(loc=0, scale=1.)
        self.mean_module = gpytorch.means.ConstantMean(prior=prior, batch_shape=torch.Size([2]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2]), ard_num_dims=EMB_SIZE),
            batch_shape=torch.Size([2])
        )
        self.embedder = embedder.to(dev)

    def cov(self, x):
        embeds = self.embedder(x)
        return self.covar_module(embeds)
        
    def forward(self, x):
        embeds = self.embedder(x)
        mean_x = self.mean_module(embeds)
        covar_x = self.covar_module(embeds)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred
    
    
class BetaGPExplorer(torch.nn.Module):
    def __init__(self, dataset: dataset.Dataset, fitter: data_fitter.Fitter, cache_dir: str, 
                 device, explore_strategy='variance', seed=0, gp_emb_size=EMB_SIZE, num_updates=50, num_ws_updates=5000, lr=1e-3, width=3, sample_type="correctedwep", freq_alpha=0, nbr=False, no_scale_loss=False, dw_alpha=0, rl_approx_type='mob'):
        """
        :param freq_alpha: sample pairs based on freq^freq_alpha
        :param nbr: sample pairs in region randomly if set to false else using (learned) kernel values.
        """
        super(BetaGPExplorer, self).__init__()
        self.dataset = dataset
        self.fitter = fitter
        self.cache_dir = cache_dir
        self.dev = device
        self.explore_strategy = explore_strategy
        self.seed = seed
        self.num_fit_updates = num_updates
        self.num_ws_updates = num_ws_updates
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.width = width
        self.lr = lr
        self.sample_type = sample_type
        print ("Sampler type: %s" % self.sample_type)
        global EMB_SIZE
        EMB_SIZE = gp_emb_size
        assert freq_alpha == 0 or nbr == False, "You cannot use both temperature scaling and kernel neighbours for specifying region synthesis!"
        self.freq_alpha = freq_alpha
        self.nbr = nbr
        self.no_scale_loss = no_scale_loss
        self.dw_alpha = dw_alpha
        self.rl_approx_type = rl_approx_type
        
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
        self.gp_model = GPClassificationModel(self.arms_tensor, self.embedder).to(self.dev)
        self.gp_likelihood = beta_gp_likelihood.BetaLikelihood(self.width, no_scale_loss=self.no_scale_loss, dw_alpha=self.dw_alpha, rloss_approx_type=self.rl_approx_type).to(self.dev)

        all_vars = list(self.gp_model.named_parameters()) + list(self.embedder.named_parameters())
        length_params = []
        all_but_length_params = []
        l_names = ["kernel.raw_lengthscale", "covar_module.raw_outputscale"]
        self.gp_params, self.embedder_params = [], []
        for name, p in self.gp_model.named_parameters():
            if name.startswith('embedder'):
                self.embedder_params.append(p)
            else:
                self.gp_params.append(p)

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
    
    def debug(self, fname):
        """
        Sanity check on the learned kernel
        Does kernel score: K(a1, a2) correspond to (negatively) the differences in value of the arms?
        """
        self.gp_model.eval()
        self.embedder.eval()
        self.gp_likelihood.eval()
        
        covmat = self.gp_model.cov(self.arms_tensor).detach().cpu().numpy()
        embs = self.gp_model.embedder(self.arms_tensor).detach().cpu().numpy()
        arm_index_to_acc = self.fitter.arm_index_to_acc
        with open(fname, "wb") as f:
            pickle.dump([covmat, embs, arm_index_to_acc], f)
    
    def unique_name(self):
        """
        Returns a string that is a unique signature of its parameters
        """
        ustr = "%s" % self.explore_strategy
        if self.sample_type != "correctedwep":
            ustr += "_st=%s" % self.sample_type

        ustr += "_width=%d" % self.width
        ustr += '_rlapprox=%s' % self.rl_approx_type
        if self.freq_alpha != 0:
            ustr += "_falpha=%0.2f" % self.freq_alpha
        if self.nbr:
            ustr += "_nbr"
        if self.no_scale_loss:
            ustr += "_nsl"
        if self.dw_alpha != 0:
            ustr += "_dwa=%0.2f" % self.dw_alpha
        return ustr
    
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
                mu = misc.mu_transform(sample[:, :, 0])
                scale = misc.scale_transform(sample[:, :, 1])
                return mu

            def variance_fn(sample):
                mu = misc.mu_transform(sample[:, :, 0])
                scale = misc.scale_transform(sample[:, :, 1])
                return mu*(1-mu)/(scale + 1)

            mean = self.quadrature(mean_fn, normal_dist)
            variance = self.quadrature(variance_fn, normal_dist)
            if debug:
                return mean, variance, torch.exp(latent_mean[:, 0]), torch.exp(latent_mean[:, 1])
            else:
                return mean, variance
    
    def fit(self, num_updates, lr): 
        self.gp_model.train()
        self.gp_likelihood.train()
        
        self.optimizer1 = torch.optim.Adam(self.gp_params + self.embedder_params, lr=lr, weight_decay=0)
        self.optimizer2 = torch.optim.Adam(self.gp_params, lr=lr, weight_decay=0)
        
        for _ in range(2):
            obs_idxs = np.where(self.explored == 1)[0]
            if _==1:
                all_idxs = np.array([idx for idx, acc in enumerate(self.fitter.arm_index_to_acc) if acc is not np.nan])
                obs_idxs = np.union1d(obs_idxs, all_idxs)
            count = (self.counts0 + self.counts1)[obs_idxs].to(self.dev)
            obs_y = self.counts1[obs_idxs].to(self.dev)

            mll = gpytorch.mlls.VariationalELBO(self.gp_likelihood, self.gp_model, obs_y.numel(), beta=1, combine_terms=True)

            cov_mat = self.gp_model.cov(self.arms_tensor[torch.where(count>0)[0]])    
            global S
            # num_regions x num_obs
            S = misc.get_random_smoothing_matrix(counts=count[torch.where(count>0)[0]].cpu().numpy(), width=self.width, cov_mat=cov_mat if self.nbr else None, alpha=self.freq_alpha)
            S /= torch.unsqueeze(S.sum(dim=-1), dim=-1)
            S = S.to(self.dev)
            
            print ("Fitting on:", len(obs_idxs), S.shape)
            if _ == 0:
                optimizer = self.optimizer1
            else:
                optimizer = self.optimizer2
            for i in tqdm.tqdm(range(num_updates)):
                optimizer.zero_grad()
                with gpytorch.settings.cholesky_jitter(1e-3):
                    output = self.gp_model(self.arms_tensor[obs_idxs])
                loss = -mll(output, obs_y, count=count, S=S, prior_acc=torch.tensor(self.prior_acc), kernel_mat=cov_mat.detach().evaluate()[0])
                loss.backward()
                optimizer.step()
        
        # debug stuff
        mean, variance, a, b = self.mean_variance(debug=True)
        mean, variance = mean.cpu(), variance.cpu()
        emp_mean = (obs_y/count).cpu()
        errs = np.abs((emp_mean - mean[obs_idxs]).numpy())
        _idx, idx = np.argmax(errs), obs_idxs[np.argmax(errs)]
        
        _c0, _c1 = self.counts0.cpu().numpy(), self.counts1.cpu().numpy()
        ds = []
        errs2 = np.abs((torch.tensor(self.fitter.arm_index_to_acc) - mean).cpu().numpy())
        bad_arms = np.argsort(-errs2)
        for ai in bad_arms:
            if self.fitter.arm_index_to_acc[ai] is not np.nan:
                ds.append("%d(%0.2f) %0.4f(%0.4f) %0.2f %0.4f" % (ai, _c1[ai]+_c0[ai], mean[ai], variance[ai], (_c1[ai]/(_c1[ai]+_c0[ai]) if (_c0[ai]+_c1[ai])>0 else np.nan), self.fitter.arm_index_to_acc[ai]))
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
                elif self.explore_strategy == 'svariance3':
                    arm_idxs = np.argsort(variance.cpu().numpy() + np.random.gumbel(0, 1, len(variance)))
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
#                 sys.stderr.write("Err of arms being explored: %s\n" % str([(ai, _c0[ai], _c1[ai], a[ai], variance[ai]) for ai in obs_arm_idxs]))
                _s1 =  np.sort(a.detach().cpu().numpy())
#                 sys.stderr.write("%s %s\n" % (_s1[-10:], _s1[:10]))
            
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
        uname = self.unique_name()
        save_name = self.cache_dir + ("/beta_gp_rloss_explore_" + uname) 
        save_name = save_name + ("_seed=%d.pkl" % self.seed)
            
        perfs = []
        print ("Total count after sampling: ", self.counts0.sum() + self.counts1.sum())
        self.fit(1000, self.lr)
        
        sys.stderr.write("Explored %d examples \n" % self.num_obs)
        metric_vals = self.eval()
        sys.stderr.write("%0.4f %0.4f\n" % (metric_vals[0], metric_vals[1]))
        perfs.append((self.num_obs, metric_vals))
        
        while self.num_obs < budget:
            self.explore(num_sample)
            self.num_obs += num_sample
            self.fit(50, self.lr)
                
            sys.stderr.write("Explored %d examples\n" % self.num_obs)
            metric_vals = self.eval()
            sys.stderr.write("%0.4f %0.4f\n" % (metric_vals[0], metric_vals[1]))
            perfs.append((self.num_obs, metric_vals))
            
            with open(save_name, "wb") as f:
                pickle.dump(perfs, f)

def estimation_ablation(args, kwargs):
    """
    Written for the purpose of ablation study of estimation and isolate the exploration
    """
    errs = {}
    
    fkwargs = {k: kwargs[k] for k in kwargs.keys() if not k.startswith('ablation')}
    
    seed = kwargs['seed']
    save_name2 = "%s/cpred_seed=%d.pkl" % (args[1].cache_dir, seed)
    exp = BetaGPExplorer(*args, **fkwargs)
    mean = [exp.prior_acc]*len(exp.counts0)
    _err = exp.fitter.evaluate_mean_estimates(mean, 0)[0]
    err2 = exp.fitter.evaluate_mean_micromse(mean)
    err3, all_err = exp.fitter.evaluate_mean_worstacc(mean)
    err4 = exp.fitter.evaluate_mean_worstfreq(mean)
    _errs = (_err, err2, err3, err4, all_err)
    print ("Variance of arms around prior", (_err, err3))
    with open(save_name2, "wb") as f:
        pickle.dump(_errs, f)
        
    ustr = exp.unique_name()
    save_name = "%s/beta_gp_rloss_explorer" % args[1].cache_dir
    save_name += "_exp=" + ustr
    save_name += "_mega_seed=%d_estimation_ablation.pkl" % seed
    
    start = 0
    if kwargs['ablation_resume']:
        with open(save_name, "rb") as f:
            errs = pickle.load(f)
            start = max(list(errs.keys())) + 200
            print ("Starting from ", start)
        
    for num_sample in np.arange(start, 3000, 100): #[500, 1500, 3000]: 
        np.random.seed(seed)
        exp = BetaGPExplorer(*args, **fkwargs)
        status = exp.fitter.sample(num_sample, exp.counts0, exp.counts1, exp.explored, sample_type=exp.sample_type)

        print ("Counts: ", exp.counts0.sum(), exp.counts1.sum())
        exp.fit(1000, 1e-2)
        err, err2, err3, err4, all_err = exp.eval()
        print ("NS: %d Seed: %d err: %f %f %f %f" % (num_sample, seed, err, err2, err3, err4))
        errs[num_sample] = (err, err2, err3, err4, all_err)
        with open(save_name, "wb") as f:
            pickle.dump(errs, f)
        
    with open(save_name, "wb") as f:
        pickle.dump(errs, f)
    return errs

def cv(args, kwargs):
    ws = [1, 3, 5]
    errs = []
    p = 0.7
    for w in ws:
        kwargs["width"] = w
        exp = BetaGPExplorer(*args, **kwargs)
        counts0, counts1 = exp.counts0.clone(), exp.counts1.clone()
        train_counts0, train_counts1 = torch.zeros_like(counts0), torch.zeros_like(counts1)
        val_counts0, val_counts1 = torch.zeros_like(counts0), torch.zeros_like(counts1)
        for i in range(len(train_counts0)):
            arr = np.random.permutation([(1, 0)]*int(counts0[i]) + [(0, 1)]*int(counts1[i]))
            _ln = int(len(arr)*p)
            if len(arr) == 1:
                if np.random.random() > p:
                    _ln = 0
                else:
                    _ln = 1
            train, val = arr[:_ln], arr[_ln:]
            for _ in train:
                train_counts0[i] += _[0]
                train_counts1[i] += _[1]
            
            for _ in val:
                val_counts0[i] += _[0]
                val_counts1[i] += _[1]
            
        print ("Train:", train_counts0.sum(), train_counts1.sum())
        print ("Val:", val_counts0.sum(), val_counts1.sum())
        exp._set_initial_counts(train_counts0, train_counts1)
        exp.fit(1000, exp.lr)
        mean, _ = exp.mean_variance()
        err = 0
        _err = []
        for i in range(len(val_counts0)):
            if val_counts0[i] + val_counts1[i] > 0:
                emp_mu = (val_counts1[i]/(val_counts0[i] + val_counts1[i])).item()
                _err.append((emp_mu - mean[i].item())**2)
        
        _err, _, _ = exp.eval()
        errs.append(_err)
        print (w, _err)
    w = ws[np.argmin(errs)]
    print ("Best width: %d errs: %s" % (w, str(errs)))
    return w