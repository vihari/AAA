import torch
import numpy as np
import gpytorch
from ..utils import misc

RLOSS_ALLOWED_APPROX = ['mob']

class BernLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    """
    These are the only two functions that are needed for ELBO computation
    """
    def __init__(self, width):
        super().__init__()
        self.width = width
        print ("Using grouping width: %s" % str(self.width))
        
    def forward(self, function_samples, **kwargs):
        """
        Instead of sampling from P_{\alpha\beta} = \sum_i w_ip_i,
        we sample from p_i and combine them with w_is, this is fine because 
        log \sum_iw_ip_i \geq \sum_i w_ilogp_i
        """
        # function_samples would be of shape 20(num samples) x batch_size 
        mu = misc.mu_transform(function_samples)
        count = kwargs["count"]
        smooth = kwargs["smooth"]
        S = kwargs["S"]
        width = self.width
        def fn(observations):
            dev = observations.device
            obs_idxs = torch.where(count>0.)[0]
            observations, obs_count = observations[obs_idxs], count[obs_idxs]
            obs_mu = mu[:, obs_idxs]
            alpha, beta = observations, obs_count - observations
            region_counts = torch.mv(S, obs_count)
            alpha, beta = torch.mv(S, alpha), torch.mv(S, beta)
            contr_S = S * torch.unsqueeze(obs_count, dim=0)
            contr_S /= torch.unsqueeze(region_counts, dim=1)

            smooth_mu = torch.mm(obs_mu, S.t())
            log_prob = alpha*torch.log(smooth_mu + 1e-5) + beta*torch.log(1 - smooth_mu + 1e-5)
            return log_prob 
        return fn
    
    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        prob_lambda = lambda function_samples: self.forward(function_samples, **kwargs)(observations)
        prior_acc = kwargs["prior_acc"]
        def prior_log_prob_fn(function_samples):
            mu = misc.mu_transform(function_samples)
            log_prob = 0.1*(prior_acc*torch.log(mu + 1e-10) + (1-prior_acc)*torch.log(1 - mu + 1e-10))
            return log_prob

        prob = self.quadrature(prob_lambda, function_dist)
        prior_log_prob = self.quadrature(prior_log_prob_fn, function_dist)
        return (prob.mean() + prior_log_prob.mean())*len(observations)