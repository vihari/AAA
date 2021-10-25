import torch
import numpy as np
import argparse

# Log-beta
def lbeta(x, y):
    return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x+y) 

##############
# Beta Parameter scaling utilities
##############

def mu_transform(x):
    return torch.clamp(torch.sigmoid(6*x - 3), min=1e-5, max=1-1e-5)

def scale_transform(x):
    return torch.clamp(torch.nn.functional.softplus(x), min=1e-3)

def a_transform(x):
    return torch.clamp(torch.nn.functional.softplus(x), min=1e-3)

def b_transform(x):
    return torch.clamp(torch.nn.functional.softplus(x), min=1e-3)

#############################
# Region Pooling Utilitities 
#############################

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
    
def get_random_smoothing_matrix(counts, width=3, cov_mat=None, alpha=0):
    """
    returns number of new obs x num_obs
    """
    assert (cov_mat == None) or (alpha==0)
    num_obs = len(counts)
    s_matrix = []
    lns = []
    if type(width) != list:
        # create 2*width number of regions of width: width
        width = [(width, 2*width)]
    for ni in range(num_obs):
        for _w, nrs in width:
            if cov_mat is None:
                p = counts**alpha
            else:
                p = cov_mat[:, ni].detach().cpu().numpy()[0] + 1e-5
            p[ni] = 0
            if counts[ni] >= _w:
                region = get_region(ni, counts, 1, p)
                s_matrix.append(region)
            for nr in range(nrs):
                region = get_region(ni, counts, _w, p)
                s_matrix.append(region)
                lns.append(len(np.nonzero(region)[0]))

    S = np.stack(s_matrix, axis=0).astype(np.float32)
    assert np.alltrue(S.sum(axis=-1)>0)   
    return torch.from_numpy(S)


##################################################################
# Command line arguments for the main files: cocos3, celeba, etc.
##################################################################
def get_arg_parser():
    parser = argparse.ArgumentParser(description='Estimatie Accuracy Surface for your task!')
    parser.add_argument('--explorer', type=str, default="bern_gp",
                        help='Explorer type: (gp, beta_gp) allowed')
    parser.add_argument('--errs', type=str, default="0.0,0.3,0.7",
                        help="Err@n at which the model is evaluated [default: 0.0,0.3,0.7]")
    parser.add_argument('--ft', type=str, default="gp",
                        help='Fit type: (gp, simple) allowed')
    parser.add_argument('--et', type=str, default="svariance2",
                        help='Explore kind: (variance, random) allowed')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--ablation', help='Ablation that removes exploration', action='store_true')
    parser.add_argument('--debug', help='Invokes debug routine in beta_gp_rloss_explorer', action='store_true')
    parser.add_argument('--ablation_resume', help='Resume ablation expt from where it failed', action='store_true', default=False)
    
    # Loss related
    parser.add_argument('--width', type=int, help='RL: Region loss width', default=3)
    parser.add_argument('--freq_alpha', type=float, help='RL: Frequency based region alpha', default=0)
    parser.add_argument('--nbr', help='RL: Make regions not random but using neighbours from DKL', action='store_true')
    parser.add_argument('--no_scale_loss', action='store_true', help="RL: Use gamma loss on counts?")
    parser.add_argument('--dw_alpha', type=float, default=0, help="RL: Scale the losses for each arm with their counts")
    parser.add_argument('--approx_type', type=str, default='mob', help="Region loss approximation type, see likelihoods/beta_gp_likelihood")
    parser.add_argument('--sample_type', type=str, default='correctedwep', help='Calibration type: allowed: ["correctedwep", "correctednoep", "raw"]')
    
    parser.add_argument('--alpha_beta', help='RL: Parameterize Beta by alpha-beta instead of mu-scale', action='store_true')
    
    return parser

def populate_params(tgt_kwargs, args):
    tgt_kwargs['width'] = args.width
    tgt_kwargs['freq_alpha'] = args.freq_alpha
    tgt_kwargs['nbr'] = args.nbr
    tgt_kwargs['no_scale_loss'] = args.no_scale_loss
    tgt_kwargs['dw_alpha'] = args.dw_alpha
    tgt_kwargs['rl_approx_type'] = args.approx_type
    tgt_kwargs['sample_type'] = args.sample_type
    if args.ablation:
        tgt_kwargs['ablation_resume'] = args.ablation_resume