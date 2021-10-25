# sanity check for if Kernel score corresponds to true value of arms
# i.e. correlation of K(a1, a2) to v(a1) - v(a2)

import pickle
import numpy as np

with open("../data/celeba/svariance2_width=3_debug.pkl", "rb") as f:
    covmat, arm_accs = pickle.load(f)
    
    idxs = np.where(not np.isnan(arm_accs))[0]
    ks = np.reshape(covmat[0, idxs, idxs], [-1])
    acc_diffs = np.reshape(np.abs(np.reshape(arm_accs[idxs], [-1, 1]) - np.reshape(arm_accs[idxs], [1, -1])), [-1])
    sidxs = np.argsort(acc_diffs)
    plt.