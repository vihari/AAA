import torch
import numpy as np
import gpytorch
from torch import distributions as distrs
from ..utils import misc

"""
Allowed region loss approximation types.
They differ in how the accuracy of the region is sampled.
mob - Mixture of Beta, \rho ~ \sum_a w_a\beta(\mu_a, \sigma_a)
mop - Mixture of paramater, \rho ~ \beta(\sum_a w_a\mu_a, \sum_a w_a\sigma_a)
tn_reparam - Sampling based exact [todo]
rignore - ignore rare arms
"""
RLOSS_ALLOWED_APPROX = ['mob', 'mop', 'tn_reparam', 'simple', 'simplev2', 'simplev3', 'baseline', 'rignore']

def rloss(rloss_approx_type, kwargs):
    """
    :param alphas, betas: Alpha and Beta params of Beta distr for each arm (number of samples x number of observed arms: nxo)
    :param region_pos_count, region_neg_count: 1 and 0 count per region (shape: r)
    :param contr_S: Contribution S number of regions x number of observed arms (rxo) -- This is S matrix normalized/weighted by contribution of each arm to the region.
    returns the logprobs for each sample and for each region (nxr)
    """
    rloss_fns = {
        'mob': mob_rloss, 
        'mop': mop_rloss, 
        'tn_reparam': truncated_normal_rloss, 
        'simple': simple_rloss,
        'simplev2': simplev2_rloss,
        'simplev3': simplev3_rloss,
        'baseline': baseline_rloss,
        'rignore': rignore_rloss
    }
    rloss_fn = rloss_fns[rloss_approx_type]
    return rloss_fn(kwargs)

def truncated_normal_rloss(kwargs):
    alpha, beta = kwargs["alpha"], kwargs["beta"]
    region_pos_count, region_neg_count = kwargs["rpc"], kwargs["rnc"]
    contr_S = kwargs["contr_S"]
    region_pos_count, region_neg_count = counts[0], counts[1]
    n_sample = 400
    mu, scale = alpha/(alpha+beta), alpha+beta
    mean, var = mu, (mu*(1-mu))/scale
    normal_distr = distrs.Normal(torch.zeros_like(mean), torch.ones_like(var))
    # n_sample x 20 x #arms
    std_normal_samples = normal_distr.sample([n_sample])
    middle_normal_samples = mean + torch.sqrt(var)*std_normal_samples

    uniform_samples = np.random.uniform(1e-3, 1, size=[n_sample, 1, 1]).astype(np.float32)
    # Approximate with exponential distribution of the same mean at the extreme
    left_exp_samples = -mean*torch.log(torch.from_numpy(uniform_samples))
    right_exp_samples = 1+mean*torch.log(torch.from_numpy(uniform_samples))

    _mu = mu.mean(dim=0).unsqueeze(dim=0)
    lsep, rsep = 0.01, 0.99
#     rhos = (_mu>=rsep).type(torch.float32)*right_exp_samples
#     rhos += ((_mu>lsep) & (_mu<rsep)).type(torch.float32)*middle_normal_samples
#     rhos += (_mu<=lsep).type(torch.float32)*left_exp_samples
    rhos = middle_normal_samples
    
    rhos = torch.clamp(rhos, min=0, max=1)
    # n_sample x 20 x #regions
    wt_mean = torch.einsum("ska,ra->skr", rhos, contr_S)
    # n_sample x 20 x #regions
    region_logll = region_pos_count*torch.log(torch.clamp(wt_mean, min=1e-5)) + region_neg_count*torch.log(torch.clamp(1 - wt_mean, min=1e-5))

    # n_sample x 20
    logprobs = region_logll.sum(dim=-1)
    # return 20 x 1
    return torch.logsumexp(logprobs, dim=0).unsqueeze(dim=-1)

def mop_rloss(kwargs):
    alpha, beta = kwargs["alpha"], kwargs["beta"]
    region_pos_count, region_neg_count = kwargs["rpc"], kwargs["rnc"]
    contr_S = kwargs["contr_S"]
    
    region_pos_count, region_neg_count = counts[0], counts[1]
    region_alpha = torch.mm(alpha, contr_S.t())
    region_beta = torch.mm(beta, contr_S.t())
    # num_samples x num_regions
    logprobs = misc.lbeta(region_alpha + region_pos_count, region_beta + region_neg_count) - misc.lbeta(region_alpha, region_beta)
    return logprobs

def simple_rloss(kwargs):
    """
    Objective = \sum_a LL(c_a+\lambda/z_a\sum_{r that includes a} c_r; \theta_a)
    z_a = \sum_{r that includes a} 1
    """
    alpha, beta = kwargs["alpha"], kwargs["beta"]
    region_pos_count, region_neg_count = kwargs["rpc"], kwargs["rnc"]
    contr_S, S = kwargs["contr_S"], kwargs["S"]
    arm_pos_count, arm_neg_count = kwargs["apc"], kwargs["anc"]
    
    lmbda = 0.2
    # Get contribution of regions to an arm -- contr_S: r x a
    _S = S
    s_pos_count = arm_pos_count + lmbda*(_S.t() @ region_pos_count)/_S.sum(dim=0)
    s_neg_count = arm_neg_count + lmbda*(_S.t() @ region_neg_count)/_S.sum(dim=0)
    logprobs = misc.lbeta(alpha + s_pos_count, beta + s_neg_count) - misc.lbeta(alpha, beta)
    return logprobs

def simplev3_rloss(kwargs):
    alpha, beta = kwargs["alpha"], kwargs["beta"]
    contr_S, S = kwargs["contr_S"], kwargs["S"]
    arm_pos_count, arm_neg_count = kwargs["apc"], kwargs["anc"]
    kernel_mat = kwargs["kernel_mat"]
    assert kernel_mat is not None
    kernel_mat = kernel_mat.detach()
    
    k = 3
    if "lmbda" not in kwargs:
        lmbda = 1./k
    else:
        lmbda = kwargs["lmbda"]
    kernel_mat /= kernel_mat.sum(dim=1).unsqueeze(dim=1)
    _tk = torch.topk(kernel_mat, k, dim=1)
    idxs, vals = _tk.indices, _tk.values
    mask = ((arm_pos_count + arm_neg_count) <= 5).type(torch.float32)
    s_pos_count, s_neg_count = arm_pos_count, arm_neg_count
    if 'rignore' not in kwargs or kwargs['rignore']:
        s_pos_count, s_neg_count = s_pos_count*(1-mask), s_neg_count*(1-mask)
    
    n = len(kernel_mat)
    for ki in range(k):
        rho = arm_pos_count/(arm_pos_count+arm_neg_count)
        num = arm_pos_count+arm_neg_count
        _idxs, _vals = idxs[torch.arange(n), ki], vals[torch.arange(n), ki]
        s_pos_count += mask*lmbda*rho[_idxs]*num[_idxs]
        s_neg_count += mask*lmbda*(1-rho[_idxs])*num[_idxs]
        
    logprobs = misc.lbeta(alpha + s_pos_count, beta + s_neg_count) - misc.lbeta(alpha, beta)
    return logprobs

def simplev2_rloss(kwargs):
    """
    for an arm a and a region r that includes it
        c_ar = c_a + \lambda c_r
        obj_ar = contr_ar LL(c_ar; \theta_a)
    """
    alphas, betas = kwargs["alpha"], kwargs["beta"]
    region_pos_count, region_neg_count = kwargs["rpc"], kwargs["rnc"]
    arm_pos_count, arm_neg_count = kwargs["apc"], kwargs["anc"]
    contr_S, S = kwargs["contr_S"], kwargs["S"]

    lmbda = 0.2
    # num active x 2
    active_ra = torch.nonzero(contr_S)
    # active
    wts = contr_S[active_ra[:, 0], active_ra[:, 1]]
    n, r, a = alphas.shape[0], contr_S.shape[0], contr_S.shape[1]
    # active
    obs_pos = arm_pos_count[active_ra[:, 1]] + lmbda*region_pos_count[active_ra[:, 0]]
    obs_neg = arm_neg_count[active_ra[:, 1]] + lmbda*region_neg_count[active_ra[:, 0]]
    # n x active
    logprobs = misc.lbeta(alphas[:, active_ra[:, 1]] + obs_pos, betas[:, active_ra[:, 1]] + obs_neg) - misc.lbeta(alphas[:, active_ra[:, 1]], betas[:, active_ra[:, 1]])
    final = torch.zeros([n, r, a], device=alphas.device)
    num_active = len(active_ra)
    # fill the matrix. repeat_interleave is [0, 1, 2] = [0, 0, 1, 1, 2, 2] when duplicity is 2.
    final[torch.arange(n).repeat_interleave(num_active, 0), active_ra[:, 0].repeat([n]), active_ra[:, 1].repeat([n])] = logprobs.flatten()
    logprobs = torch.log(torch.clamp(torch.einsum("nra,ra->nr", torch.exp(final), S), min=1e-30))
    return logprobs
    
def baseline_rloss(kwargs):
    kwargs["lmbda"] = 0
    kwargs["rignore"] = False
    return simplev3_rloss(kwargs)

def rignore_rloss(kwargs):
    kwargs['rignore'] = True
    kwargs['lmbda'] = 0
    return simplev3_rloss(kwargs)
    
def mob_rloss(kwargs):
    """
    This code is 6x faster than 'mob_slow_rloss'
    """
    alphas, betas = kwargs["alpha"], kwargs["beta"]
    region_pos_count, region_neg_count = kwargs["rpc"], kwargs["rnc"]
    contr_S = kwargs["contr_S"]

    # num active x 2
    active_ra = torch.nonzero(contr_S)
    # active
    wts = contr_S[active_ra[:, 0], active_ra[:, 1]]
    n, r, a = alphas.shape[0], contr_S.shape[0], contr_S.shape[1]
    # n x active
    logprobs = misc.lbeta(alphas[:, active_ra[:, 1]] + region_pos_count[active_ra[:, 0]], betas[:, active_ra[:, 1]] + region_neg_count[active_ra[:, 0]]) - misc.lbeta(alphas[:, active_ra[:, 1]], betas[:, active_ra[:, 1]])
    final = torch.zeros([n, r, a], device=alphas.device)
    num_active = len(active_ra)
    # fill the matrix. repeat_interleave is [0, 1, 2] = [0, 0, 1, 1, 2, 2] when duplicity is 2.
    final[torch.arange(n).repeat_interleave(num_active, 0), active_ra[:, 0].repeat([n]), active_ra[:, 1].repeat([n])] = logprobs.flatten()
    # Below is 2x slower than the above line.
    # for ni in range(n):
    #    final[ni, active_ra[:, 0], active_ra[:, 1]] = logprobs[ni]
    logprobs = torch.log(torch.clamp(torch.einsum("nra,ra->nr", torch.exp(final), contr_S), min=1e-30))
    return logprobs

def mob_slow_rloss(kwargs):
    """
    Do not use, many redundant computations
    """
    alphas, betas = kwargs["alpha"], kwargs["beta"]
    region_pos_count, region_neg_count = kwargs["rpc"], kwargs["rnc"]
    contr_S = kwargs["contr_S"]
    if True:
        _alphas, _betas = alphas.unsqueeze(-1), betas.unsqueeze(-1)
        # n x o x r
        logprobs = misc.lbeta(_alphas+region_pos_count, _betas+region_neg_count) - misc.lbeta(_alphas, _betas)
        logprobs = torch.log(torch.clamp(torch.einsum("nor,or->nr", torch.exp(logprobs), contr_S.t()), min=1e-30))
        return logprobs
    else:
        # No speed-up from explicit integration, infact slightly slower 
        # and huge memory footprint however, got slightly better results 
        # from doing that, possible I was only fooled by randomness
        # no extremes, no nans
        int_theta = torch.arange(1e-2, 1-1e-2, 5e-2).view([1, 1, -1]).to(alphas.device)
        beta_distrs = torch.distributions.Beta(alphas.unsqueeze(-1), betas.unsqueeze(-1))
        # 20 x #obs x #int_theta
        logprobs = beta_distrs.log_prob(int_theta)
        # logZ: 20 x #obs x 1 
        logZ = logprobs.logsumexp(dim=-1).unsqueeze(-1)
        # log p_{\alpha\beta}
        logprobs = logprobs - logZ

        # gearing up int_theta for many regions, the last dimension is for the regions
        int_theta = int_theta.view([-1, 1])
        region_pos_count, region_neg_count = region_pos_count.view([1, -1]), region_neg_count.view([1, -1])
        # #int_theta x num_regions
        region_logprobs = region_pos_count*torch.log(int_theta) + region_neg_count*torch.log(1 - int_theta)

        # carry the integration and then the weighted sum
        # 20 x #obs x #int_theta x #regions
        probs = torch.exp(region_logprobs.view(1, 1, len(int_theta), -1)) * torch.exp(logprobs.unsqueeze(-1))
        # 20 x #obs x #regions
        probs = probs.sum(dim=2)
        # weighted sum
        # 20 x #regions
        logprobs = torch.log(torch.einsum("nor,or->nr", probs, contr_S.t()))

        return logprobs 

class BetaLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    """
    Beta (mu-scale)
    """
    def __init__(self, width, rloss_approx_type='simple', no_scale_loss=False, dw_alpha=0):
        super().__init__()
        self.width = width
        assert rloss_approx_type in RLOSS_ALLOWED_APPROX, "Approximation type should be one of %s" % ALLOWED_APPROX
        self.rloss_approx_type = rloss_approx_type
        self.no_scale_loss = no_scale_loss
        self.dw_alpha = dw_alpha
    
    def forward(self, function_samples, **kwargs):
        # function_samples would be of shape 20(num samples) x batch_size x 2(num_tasks) 
        logmu, logscale = function_samples[:, :, 0], function_samples[:, :, 1]
        count = kwargs["count"]
        S = kwargs["S"]
        kernel_mat = kwargs["kernel_mat"]
        mu, scale = misc.mu_transform(logmu), misc.scale_transform(logscale)
        obs_idxs = torch.where(count>0)[0]
        mu, scale = mu[:, obs_idxs], scale[:, obs_idxs]
        def fn(observations):
            observations, obs_count = observations[obs_idxs], count[obs_idxs]
            dev = observations.device
            
            pos_count, neg_count = observations, obs_count - observations
            region_counts = torch.mv(S, obs_count)
            contr_S = S * torch.unsqueeze(obs_count, dim=0)
            contr_S /= torch.unsqueeze(region_counts, dim=1)
            
            region_pos_count, region_neg_count = torch.mv(S, pos_count), torch.mv(S, neg_count)
            alphas, betas = mu*scale, (1-mu)*scale
            counts = [region_pos_count, region_neg_count, pos_count, neg_count]
            kw_args = {'rpc': region_pos_count, 'rnc': region_neg_count, 
                       'apc': pos_count, 'anc': neg_count, 'contr_S': contr_S, 'S': S, 
                       'alpha': alphas, 'beta': betas, 'kernel_mat': kernel_mat}
            _rloss = rloss(self.rloss_approx_type, kw_args)
            if not self.no_scale_loss:
                # scale -- n x o
                g = torch.distributions.gamma.Gamma(scale, 1)
                # n x o
                scale_lprobs = g.log_prob(torch.clamp(obs_count, min=1e-2, max=10))
                # n x 1
                scale_lprobs = scale_lprobs.mean(dim=-1).unsqueeze(dim=-1)
                # scale_lprobs = torch.log(torch.einsum("nor,or->nr", torch.exp(scale_lprobs), contr_S.t()))
                _rloss = _rloss + scale_lprobs
            
            if self.dw_alpha > 0:
                dwt = obs_count**(self.dw_alpha)
                dwt /= dwt.sum(dim=0)
                assert len(dwt.shape)==1
                _rloss *= dwt
            return _rloss
        return fn
    
    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        logprob_lambda = lambda function_samples: self.forward(function_samples, **kwargs)(observations)
        prior_acc = kwargs["prior_acc"]
        def prior_log_prob_fn(function_samples):
            logmu, logscale = function_samples[:, :, 0], function_samples[:, :, 1]
            mu, scale = misc.mu_transform(logmu), misc.scale_transform(logscale)
            alphas, betas = mu*scale, (1-mu)*scale
            log_prob = misc.lbeta(alphas + 0.1*prior_acc, betas + 0.1*(1-prior_acc)) - misc.lbeta(alphas, betas)
            # Torch-1.7 got rid of BetaBinomial. Pytorch, why would you do that?
            # log_prob = gpytorch.distributions.base_distributions.BetaBinomial(alphas, betas, 0.1).log_prob(torch.tensor(prior_acc))
            return log_prob
        
        logprob = self.quadrature(logprob_lambda, function_dist)
        if prior_acc is not None:
            prior_log_prob = self.quadrature(prior_log_prob_fn, function_dist)
            logprob = (logprob.mean() + prior_log_prob.mean())*len(observations)
        else:
            logprob = logprob.sum()
        return logprob

class BetaABLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    """
    Beta (alpha, beta)
    """
    def __init__(self, width, rloss_approx_type='baseline', no_scale_loss=None, dw_alpha=None):
        super().__init__()
        assert rloss_approx_type in RLOSS_ALLOWED_APPROX, "Approximation type should be one of %s" % ALLOWED_APPROX
        self.rloss_approx_type = rloss_approx_type
        self.width = width
            
    def forward(self, function_samples, **kwargs):
        # function_samples would be of shape 20(num samples) x batch_size x 2(num_tasks)
        count = kwargs["count"]
        S = kwargs["S"]
        kernel_mat = kwargs["kernel_mat"]
        alphas, betas = misc.a_transform(function_samples[:, :, 0]), misc.b_transform(function_samples[:, :, 1])
        obs_idxs = torch.where(count>0)[0]
        alphas, betas = alphas[:, obs_idxs], betas[:, obs_idxs]
        def fn(observations):
            observations, obs_count = observations[obs_idxs], count[obs_idxs]
            dev = observations.device
            
            pos_count, neg_count = observations, obs_count - observations            
            region_pos_count, region_neg_count = torch.mv(S, pos_count), torch.mv(S, neg_count)
            region_counts = region_pos_count + region_neg_count
            contr_S = S * torch.unsqueeze(obs_count, dim=0)
            contr_S /= torch.unsqueeze(region_counts, dim=1)
            
            kw_args = {'rpc': region_pos_count, 'rnc': region_neg_count, 
                       'apc': pos_count, 'anc': neg_count, 'contr_S': contr_S, 'S': S, 
                       'alpha': alphas, 'beta': betas, 'kernel_mat': kernel_mat}
            _rloss = rloss(self.rloss_approx_type, kw_args)
            return _rloss
        return fn
    
    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        logprob_lambda = lambda function_samples: self.forward(function_samples, **kwargs)(observations)
        prior_acc = kwargs["prior_acc"]
        def prior_log_prob_fn(function_samples):
            alphas, betas = misc.a_transform(function_samples[:, :, 0]), misc.b_transform(function_samples[:, :, 1])
            log_prob = misc.lbeta(alphas + 0.1*prior_acc, betas + 0.1*(1-prior_acc)) - misc.lbeta(alphas, betas)
            return log_prob

        logprob = self.quadrature(logprob_lambda, function_dist)
        prior_log_prob = self.quadrature(prior_log_prob_fn, function_dist)
        return (logprob.mean() + prior_log_prob.mean())*len(observations)