import pickle
import numpy as np
            
def estimation_ablation(task, idx=None, ctype='all'):
    print ("Task: %s" % task)
    if ctype == 'all':
        ckpt_names = [
            "cpred_seed=%d.pkl",
            "simple_exp=svariance2_seed=%d_estimation_ablation.pkl",
            "bern_gp_explorer_width=1_seed=%d_estimation_ablation.pkl",
            "beta_gp_rloss_explorer_exp=svariance2_width=3_rlapprox=baseline_nsl_seed=%d_estimation_ablation.pkl",
            "beta_gp_rloss_explorer_exp=svariance2_width=3_rlapprox=baseline_seed=%d_estimation_ablation.pkl",
    #         "beta_gp_rloss_explorer_exp=svariance2_width=3_rlapprox=rignore_seed=%d_estimation_ablation.pkl",
            "beta_gp_rloss_explorer_exp=svariance2_width=3_rlapprox=simplev3_nbr_seed=%d_estimation_ablation.pkl",
    #         "beta_gp_rloss_explorer_exp=svariance2_width=1_rlapprox=baseline_dwa=1.00_seed=%d_estimation_ablation.pkl"
        ]
    elif ctype == 'ab':
        ckpt_names = [
            "beta_gp_rloss_explorer_exp=svariance2_width=3_rlapprox=baseline_nsl_seed=%d_estimation_ablation.pkl",
            "betaab_gp_rloss_explorer_exp=svariance2_width=3_rlapprox=baseline_nsl_seed=%d_estimation_ablation.pkl",
        ]
    elif ctype == 'stability':
        ckpt_names = [
            "beta_gp_rloss_explorer_exp=svariance2_width=3_rlapprox=baseline_nsl_seed=%d_estimation_ablation.pkl",
            "beta_gp_rloss_explorer_exp=svariance2_width=3_rlapprox=baseline_nsl_3000lr1e-2_seed=%d_estimation_ablation.pkl",
        ]
    elif ctype == 'check':
        """
        Prior by default is imposed individually, it should ideally be imposed by just summing them to the counts.
        Which is what is being checked. Actually, theyare both the same.
        """
        ckpt_names = [
            "beta_gp_rloss_explorer_exp=svariance2_width=3_rlapprox=baseline_seed=%d_estimation_ablation.pkl",
            "beta_gp_rloss_explorer_exp=svariance2_width=3_rlapprox=baseline_check_seed=%d_estimation_ablation.pkl",
        ]
    elif ctype == 'mega':
        """
        Just that these models are evaluated for sizes: np.arange(0, 3000, 100)
        """
        ckpt_names = [
            "beta_gp_rloss_explorer_exp=svariance2_width=3_rlapprox=baseline_nsl_mega_seed=%d_estimation_ablation.pkl",
            "beta_gp_rloss_explorer_exp=svariance2_width=3_rlapprox=simplev3_nbr_mega_seed=%d_estimation_ablation.pkl",
        ]
        
    ckpt_names = ["data/" + task + "/"+ ckpt_name for ckpt_name in ckpt_names]
    def m_s(ckpt_name, idx):
        m_s_dict = {}
        for seed in range(3):
            ckpt_name_wseed = ckpt_name % seed
            with open(ckpt_name_wseed, "rb") as f:
                x = pickle.load(f)
                if ckpt_name.find('cpred') >= 0:
                    x = {500: x}
                for num in x.keys():
                    m_s_dict[num] = m_s_dict.get(num, []) + [x[num][:-1]]
        for num in m_s_dict.keys():
            _arr = np.stack(m_s_dict[num], axis=0)
            assert len(_arr) == 3, "ckpt_name: %s" % ckpt_name
            m_s_dict[num] = (np.mean(_arr, axis=0)[idx], np.std(_arr, axis=0)[idx])
        return m_s_dict
     
    def cmetric(err_dict):
        arr = [(_["diff"], _["true"]) for _ in err_dict]
        sarr = sorted(arr, key=lambda _: _[1])
        num = 50
        x = np.mean([_d**2 for _d, _t in sarr[:num]])
        return x
    
    def custom(ckpt_name):
        m_s_dict = {}
        for seed in range(3):
            ckpt_name_wseed = ckpt_name % seed
            with open(ckpt_name_wseed, "rb") as f:
                x = pickle.load(f)
                if ckpt_name.find('cpred') >= 0:
                    x = {500: x}
                for num in x.keys():
                    _m = cmetric(x[num][-1])
                    m_s_dict[num] = m_s_dict.get(num, []) + [_m]
        for num in m_s_dict.keys():
            _arr = np.stack(m_s_dict[num], axis=0)
            assert len(_arr) == 3, "ckpt_name: %s" % ckpt_name
            m_s_dict[num] = (np.mean(_arr, axis=0)[idx], np.std(_arr, axis=0)[idx])
        return m_s_dict
        
    if idx is None:
        m_s_dicts = [m_s(ckpt_name, 0) for ckpt_name in ckpt_names]
        worst_m_s_dicts = [custom(ckpt_name) for ckpt_name in ckpt_names]
        # worst_m_s_dicts = [m_s(ckpt_name, 2) for ckpt_name in ckpt_names]
    else:
        m_s_dicts = [m_s(ckpt_name, idx) for ckpt_name in ckpt_names]
    for wi in [500, 1500, 3000]:
        num = wi
        out_str = "|%d | " % (num,)
        for mi, m_s_dict in enumerate(m_s_dicts):
            m, s = np.nan, np.nan
            if wi in m_s_dict:
                _ = m_s_dict[wi]
                m, s = _[0], _[1]
                if idx is None:
                    s, _ = list(worst_m_s_dicts[mi][wi])
            
            out_str += ("%0.1f / %0.1f" % (m*100, s*100))
            if mi < len(m_s_dicts) - 1:
                out_str += " | "
        print (out_str + "|")  


if __name__ == '__main__':
#     for idx in [0, 1, 2, 3]:
#         print ("\n\nIdx: %d\n" % idx)
    ctype = 'check'
    idx = None
    if ctype == 'all':
        print ("|      |CPred|Simple|BernGP|BetaGP|BetaGP-Sl|BetaGP-Sl-Pool|")
    elif ctype == 'ab':
        idx = 0
        print ("|      |mu-scale|a-b|")
    elif ctype == 'stability':
        print ("       |short|long|")
    elif ctype == 'check':
        print ("       |default|check|")
    for d in ['cocos3_10k', 'cocos3', 'celeba_private', 'celeba']:
        estimation_ablation(d, idx=idx, ctype=ctype)
        