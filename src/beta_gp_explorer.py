import torch
from torch import distributions as distrs
import numpy as np
import sys
import tqdm
import pickle
import torch.nn.functional as F
import gpytorch

from . import dataset, data_fitter
from .utils import config
from . import beta_gp_rloss_explorer

if config.xla:
    import torch_xla.core.xla_model as xm

# https://github.com/pytorch/pytorch/blob/master/torch/distributions/kl.py
def _kl_beta_beta(p, q):
    p_concentration0, p_concentration1 = p
    q_concentration0, q_concentration1 = q
    sum_params_p = p_concentration1 + p_concentration0
    sum_params_q = q_concentration1 + q_concentration0
    t1 = q_concentration1.lgamma() + q_concentration0.lgamma() + (sum_params_p).lgamma()
    t2 = p_concentration1.lgamma() + p_concentration0.lgamma() + (sum_params_q).lgamma()
    t3 = (p_concentration1 - q_concentration1) * torch.digamma(p_concentration1)
    t4 = (p_concentration0 - q_concentration0) * torch.digamma(p_concentration0)
    t5 = (sum_params_q - sum_params_p) * torch.digamma(sum_params_p)
    return t1 - t2 + t3 + t4 + t5
    
class BetaExplorer(torch.nn.Module):
    """
    The training slows down with training. After much effort, I could not find any issue with the code.
    The slowdown could be because there are more counts observed meaning more non-zero entries in count fitting. 
    Also, as training progresses smoothness constraint is challenged.  
    """
    def __init__(self, dataset: dataset.Dataset, fitter: data_fitter.Fitter, cache_dir: str, device, 
                 explore_strategy='variance', err_ns=[.03, .05, .1]):
        super(BetaExplorer, self).__init__()
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
        self._init_params()
        self.num_obs = 0
    
        linear_layer = torch.nn.Linear(20, 10)
        torch.nn.init.normal_(linear_layer.weight, 0, 1e-3)
        self.embedder = torch.nn.Sequential(
            self.fitter.kernel_model.embedding_model,
            torch.nn.ReLU(),
            linear_layer,
        ).to(self.dev)
        self.optimizer = torch.optim.Adam(lr=1e-3, params=list(self.embedder.parameters()) + [self.logalphas, self.logbetas])
#         self.optimizer = torch.optim.SGD(lr=1e-1, params=list(self.embedder.parameters()) + [self.logalphas, self.logbetas])
        self.arms_tensor = torch.from_numpy(self.dataset.arms).type(torch.float32).to(self.dev)
#         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=10))
        self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=10)
        self.simple_pairwise_kernel = gpytorch.kernels.RBFKernel()

    def laplace_smooth_counts(self, avg_acc):
        # laplacian smothing for counts
        laplace_alpha = 0.1
        self.counts0 += torch.ones([self.dataset.num_arms], device=self.dev)*laplace_alpha*((1-avg_acc)/avg_acc)
        self.counts1 += torch.ones([self.dataset.num_arms], device=self.dev)*laplace_alpha

    def _init_params(self):
        self.logalphas = torch.nn.Parameter(-1*torch.ones([self.num_arms], device=self.dev))
        self.logbetas = torch.nn.Parameter(-1*torch.ones([self.num_arms], device=self.dev))
#         self.fitter.kernel_model.load_state_dict(self.init_state_dict)
        
    @property
    def alphas(self):
        return torch.exp(self.logalphas)
    
    @property
    def betas(self):
        return torch.exp(self.logbetas)

    def step_loss(self):
        batch_size = 256
        batch_idxs = np.random.choice(len(self.dataset.arms), batch_size, replace=False)
#         batch_idxs = np.arange(len(self.dataset.arms))
            
        alphas, betas = self.alphas[batch_idxs], self.betas[batch_idxs]
        alphas_helper, betas_helper = torch.unsqueeze(self.alphas[batch_idxs], dim=1), torch.unsqueeze(self.betas[batch_idxs], dim=1)

        embeds = self.embedder(self.arms_tensor[batch_idxs])
        n = len(embeds)
        covar_lazytensor = self.covar_module(embeds)
        K_rho = covar_lazytensor.evaluate()
        if np.random.random() < 0.01:
            print (K_rho)

        # nxn
#         kl_divs = distrs.kl_divergence(beta_distrs_helper, beta_distrs)
        kl_divs = _kl_beta_beta((alphas_helper, betas_helper), (alphas, betas))
        assert list(kl_divs.shape) == [n, n], "Unexpected size of KL div. expected: %s, found: %s" % ([n, n], kl_divs.shape)
        smoothness_loss = torch.sum(K_rho * kl_divs)/len(batch_idxs) #/len(batch_idxs)
        _counts0, _counts1 = self.counts0[batch_idxs], self.counts1[batch_idxs]
        _alphas, _betas = self.alphas[batch_idxs], self.betas[batch_idxs]
        fit_lprob = torch.lgamma(_counts0 + _alphas) + torch.lgamma(_counts1 + _betas) + torch.lgamma(_alphas + _betas)
        fit_lprob -= (torch.lgamma(_alphas) + torch.lgamma(_betas) + torch.lgamma(_counts0 + _alphas + _counts1 + _betas))

        fit_lprob = torch.sum(fit_lprob) #/self.num_obs
        det = 0.0001*gpytorch.logdet(covar_lazytensor)

#         del alphas_helper, betas_helper
        loss = smoothness_loss - fit_lprob + det
        return loss, smoothness_loss, fit_lprob, det, torch.trace(K_rho) - torch.sum(K_rho)
    
    def step_loss2(self, nstep):
        batch_size = 100
        batch_idxs = np.random.choice(len(self.dataset.arms), batch_size)
            
        alphas, betas = self.alphas[batch_idxs], self.betas[batch_idxs]
        mode = betas/(alphas+betas)
        beta = distrs.Beta(alphas, betas)
        mode_prob = torch.exp(beta.log_prob(mode))
        # nxn matrix of importance -- outer product
        pairwise_importance = torch.ger(mode_prob, mode_prob)
        # nxn
        arm_div = (mode.unsqueeze(-1) - mode.unsqueeze(0))**2
        # arm_div = self.simple_pairwise_kernel(mode, mode).evaluate()
        
        embeds = self.embedder(self.arms_tensor[batch_idxs])
        n = len(embeds)
        covar_lazytensor = self.covar_module(embeds)
        K_rho = covar_lazytensor.evaluate()
        np_k_rho = K_rho.detach().cpu().numpy()
        if nstep > 1000:
            for bi in range(batch_size):
                for bj in range(batch_size):
                    if bi == bj:
                        continue
                    if np_k_rho[bi, bj] > 0.99:
                        print ("Embeds: %s %s corr to: %d %d are very close" % (embeds[bi], embeds[bj], bi, bj))
        
        smoothness_loss = (arm_div*pairwise_importance*K_rho).sum()
        _counts0, _counts1 = self.counts0[batch_idxs], self.counts1[batch_idxs]
        _alphas, _betas = self.alphas[batch_idxs], self.betas[batch_idxs]
        fit_lprob = torch.lgamma(_counts0 + _alphas) + torch.lgamma(_counts1 + _betas) + torch.lgamma(_alphas + _betas)
        fit_lprob -= (torch.lgamma(_alphas) + torch.lgamma(_betas) + torch.lgamma(_counts0 + _alphas + _counts1 + _betas))

        fit_lprob = torch.sum(fit_lprob) #/self.num_obs
#         det = torch.diag(covar_lazytensor.evaluate()).prod()
        det = gpytorch.logdet(covar_lazytensor)
#         det = torch.tensor(0)

        if np.random.random() < 0.01:
            print (K_rho, det)

#         del alphas_helper, betas_helper
        loss = smoothness_loss - fit_lprob
        return loss, smoothness_loss, fit_lprob, det, torch.trace(K_rho) - torch.sum(K_rho)
    
    def fit(self):        
        nsteps = 2000
        log_det_coeff = 0.01
        batch_size = 512
        for nstep in tqdm.tqdm(range(nsteps), desc='Fitting soft'):
            loss, smoothness_loss, fit_lprob, log_det, _ = self.step_loss2(nstep)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if not str(self.dev).startswith('tpu'):
                self.optimizer.step()
            else:
                xm.optimizer_step(self.optimizer, barrier=True)
            
            if nstep%100 == 0:
                sys.stderr.write ("Loss: %0.4f (%0.4f %0.4f %0.4f %0.4f)\n" % (loss.detach().cpu().item(), smoothness_loss.detach().cpu().item(), fit_lprob.detach().cpu().item(), log_det.detach().cpu().item(), _.detach().cpu().item()))
        sys.stderr.write("Fitted betas on %d, %d positive and total count\n" % (self.counts1.sum().item(), (self.counts1 + self.counts0).sum().item()))
        
        _c0, _c1 = self.counts0.cpu().numpy(), self.counts1.cpu().numpy()
        _e = (self.betas/(self.alphas + self.betas)).detach().cpu().numpy()
        err, err_obs = 0, []
        for si in range(self.num_arms):
            if (_c0[si]+_c1[si]) > 0:
                true = _c1[si]/(_c0[si] + _c1[si])
                estimate = _e[si]
                err_obs += [np.abs(true-estimate)]

        print ("Length of params: ", len(list(self.parameters())))
        print ("Errs: mean, median, max: ", np.mean(err_obs), np.median(err_obs), np.max(err_obs))

    def observe(self, arm_ids, corrs):
        """
        Update the counts0 and counts1
        :arm_ids: list of arm indices
        :corrs: list of corrs: [{0, 1}]
        """
        corrs = np.array(corrs).astype(np.int32)
        total_corr = 0
        for aid, corr in zip(arm_ids, corrs):
            self.counts0[aid] += (1 - corr)
            self.counts1[aid] += corr
            total_corr += corr
        self.num_obs += len(arm_ids)
        sys.stderr.write("Observed %d examples\n" % len(corrs))
        sys.stderr.write("Total examples seen: %d\n" % (self.counts0 + self.counts1).sum().item())
    
    def explore(self, num_explore):
        with torch.no_grad():
            beta_distr = distrs.Beta(self.alphas, self.betas)
            if self.explore_strategy == 'variance':
                arm_idxs = torch.argsort(-beta_distr.variance)
                arm_idxs = arm_idxs.cpu().numpy()
            elif self.explore_strategy == 'random':
                arm_idxs = np.arange(self.num_arms)
                np.random.shuffle(arm_idxs)
            elif self.explore_strategy == 'underfit':
                means = (self.betas/(self.alphas+self.betas)).detach().cpu().numpy()
                _c0, _c1 = self.counts0.cpu().numpy(), self.counts1.cpu().numpy()
                _c = _c0+_c1
                errs = []
                for ai in range(self.num_arms):
                    if _c[ai] > 0:
                        errs.append(np.abs(_c1[ai]/_c[ai] - means[ai]))
                    else:
                        errs.append(-1)
                arm_idxs = np.argsort(-errs)
        
            x, y = [], []
            obs_arm_idxs = []
            for arm_idx in arm_idxs:
                status = self.fitter.sample_arm(arm_idx, num_sample=1, counts0=self.counts0, counts1=self.counts1)
                if status:
                    obs_arm_idxs.append(arm_idx)
                if len(obs_arm_idxs) == num_explore:
                    break

            _c0, _c1 = self.counts0.cpu().numpy(), self.counts1.cpu().numpy()
            a, b = self.alphas.detach().cpu().numpy(), self.betas.detach().cpu().numpy()
            sys.stderr.write("Err of arms being exploreds: %s\n" % str([(ai, _c0[ai], _c1[ai], a[ai], b[ai]) for ai in obs_arm_idxs]))

    def eval(self):
        mu_hat = (self.betas/(self.alphas+self.betas)).detach().cpu().numpy()
        err_1, all_err = self.fitter.evaluate_mean_estimates(mu_hat, self.err_ns[0], debug=True)
        err_2, err_3 = [self.fitter.evaluate_mean_estimates(mu_hat, _)[0] for _ in self.err_ns[1:]]
        
        bad_arms = np.argsort(-np.array(all_err))
        _c0, _c1 = self.counts0.cpu().numpy(), self.counts1.cpu().numpy()
        a, b = self.alphas.detach().cpu().numpy(), self.betas.detach().cpu().numpy()
        debug_str = str(["%d %0.4f %0.4f %0.3f %0.3f %0.3f %0.3f" % (ai, all_err[ai], self.fitter.arm_index_to_acc[ai], _c0[ai], _c1[ai], a[ai], b[ai]) for ai in bad_arms[:50]])
        sys.stderr.write("Worst arms: %s\n" % debug_str)
        
        return err_1, err_2, err_3
    
    def constant_predictor(self, num_sample=2000, corrected=True):
        counts0, counts1 = np.zeros([self.dataset.num_arms]), np.zeros([self.dataset.num_arms])
        status = self.fitter.sample(num_sample, counts0, counts1, corrected=corrected)
        acc = np.sum(counts1)/(np.sum(counts0) + np.sum(counts1))
        sys.stderr.write("Empirical accuracy: %0.4f\n" % acc)
        err_1, err_5, err_10 = [self.fitter.evaluate_mean_estimates([acc]*self.num_arms, _)[0] for _ in [0.01, 0.05, 0.1]]
        sys.stdout.write("Constant predictors perf.: %0.4f %0.4f %0.4f\n" % (err_3, err_5, err_10))
        return err_1, err_5, err_10
        
    def brute_predictor(self, num_sample=2000, sample_type="correctedwep", smooth=True):
        counts0, counts1 = np.zeros([self.dataset.num_arms]), np.zeros([self.dataset.num_arms])
        self.fitter.warm_start_counts(counts0, counts1)
        prior_acc = counts1.sum()/(counts0.sum() + counts1.sum())
                    
        status = self.fitter.sample(num_sample, counts0, counts1, sample_type=sample_type)
        print ("NS: %d, seed: %d" % (num_sample, self.dataset.seed), counts0.sum(), counts1.sum())
        acc_per_bucket = []
        num_untouched = 0
        for ai in range(self.num_arms):
            if counts0[ai] + counts1[ai] == 0:
                _acc = prior_acc
                num_untouched += 1
            else:
                n = 0.1
                _acc = (counts1[ai] + prior_acc*n)/(counts0[ai] + counts1[ai] + n)
            acc_per_bucket.append(_acc)
            
        print ("Num untouched: %d/%d" % (num_untouched, self.num_arms))
        err_1, all_err = self.fitter.evaluate_mean_estimates(acc_per_bucket, self.err_ns[0], debug=True)
        err_2, err_3 = [self.fitter.evaluate_mean_estimates(acc_per_bucket, thresh)[0] for thresh in self.err_ns[1:]]
        sys.stdout.write("Brute force predictor perf.: %0.4f %0.4f %0.4f\n" % (err_1, err_2, err_3))
        return err_1, err_2, err_3, all_err
    
    def brute_predictor_wpool(self, num_sample=2000, width=3, sample_type="correctedwep", smooth=True):
        counts0, counts1 = np.zeros([self.dataset.num_arms]), np.zeros([self.dataset.num_arms])
        self.fitter.warm_start_counts(counts0, counts1)
        prior_acc = counts1.sum()/(counts0.sum() + counts1.sum())

        status = self.fitter.sample(num_sample, counts0, counts1, sample_type=sample_type)
        print ("NS: %d, seed: %d" % (num_sample, self.dataset.seed), counts0.sum(), counts1.sum())
        n = 0.1
        counts1 += prior_acc*n
        counts0 += (1-prior_acc)*n
        A = beta_gp_rloss_explorer.get_random_smoothing_matrix(counts0+counts1, width).numpy()
        region_acc = (A @ (counts1)) / (A @ (counts0 + counts1))
        A1 = A * np.expand_dims(counts0 + counts1, axis=0)
        A1 /= np.expand_dims(A1.sum(axis=-1), axis=-1)
        rho = np.linalg.pinv(A1, rcond=1e-3) @ region_acc
        assert len(rho) == self.dataset.num_arms
        
        err_1, all_err = self.fitter.evaluate_mean_estimates(rho, self.err_ns[0], debug=True)
        err_2, err_3 = [self.fitter.evaluate_mean_estimates(rho, thresh)[0] for thresh in self.err_ns[1:]]
        sys.stdout.write("Brute force predictor perf.: %0.4f %0.4f %0.4f\n" % (err_1, err_2, err_3))
        return err_1, err_2, err_3, all_err
    
    def explore_and_fit(self, budget=5000):
        ws_num = 500
        num_sample = 50
        
        save_name = self.cache_dir + ("/%s_explore_heavyshare_perf.pkl" % self.explore_strategy) 
        perf = []
        # draw 100 examples randomly and fit
        print ("Total count before sampling: ", self.counts0.sum() + self.counts1.sum())
        status = self.fitter.sample(ws_num, self.counts0, self.counts1)
        self.num_obs += ws_num
        print ("Total count after sampling: ", self.counts0.sum() + self.counts1.sum())
        self.laplace_smooth_counts(self.counts1.sum()/(self.counts0.sum()+self.counts1.sum()))
        self.fit()
        
        sys.stderr.write("Explored %d examples \n" % self.num_obs)
        max_err, mean_err, hard_err = self.eval()
        sys.stderr.write("%0.4f %0.4f %0.4f\n" % (max_err, mean_err, hard_err))                
        perf.append((self.num_obs, max_err, mean_err, hard_err))
        
        while self.num_obs<budget:
            self.explore(num_sample)
            self.num_obs += num_sample
            self.fit()

            sys.stderr.write("Explored %d examples\n" % self.num_obs)
            max_err, mean_err, hard_err = self.eval()
            sys.stderr.write("%0.4f %0.4f %0.4f\n" % (max_err, mean_err, hard_err))        
            perf.append((self.num_obs, max_err, mean_err, hard_err))
            
            with open(save_name, "wb") as f:
                pickle.dump(perf, f)
                
def estimation_ablation(args, kwargs):
    """
    Written for the purpose of ablation study of estimation and isolate the exploration
    """
    errs = {}
    for num_sample in [500, 1500, 3000]: 
        _errs = []
        for seed in range(3):
            kwargs['seed'] = seed
            exp = BernGPExplorer(*args, **kwargs)
            status = exp.fitter.sample(num_sample, exp.counts0, exp.counts1, exp.explored)
            
            print (exp.counts0.sum(), exp.counts1.sum())
            exp.fit(5000)
            err, _, _ = exp.eval()
            _errs.append(err)
        m, s = np.mean(_errs), np.std(_errs)
        print (m, s)
        errs[num_sample] = (m, s)
        
    with open("%s/bern_gp_explorer2_estimation_ablation.pkl" % exp.cache_dir, "wb") as f:
        pickle.dump(errs, f)
    return errs