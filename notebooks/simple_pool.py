import os, sys
import torch
import pickle
import numpy as np

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.utils import train_utils, config, data_utils
from src import dataset, data_fitter, beta_explorer
import cocos3, celeba

from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 20})
def plot(task, title, save_fname, legend=False, mark=False):
    cals = [("cal:full", "r*-", "Cal:Full"), ("cal:temp", "b*-", "Cal:Temp"), ("cal:raw", "k*-", "Cal:Raw")]
    for cal, fmt, name in cals:
        with open("../data/%s/%s.pkl" % (task, cal), "rb") as f:
            perfs = pickle.load(f)
            xs, aucs, _, _ = zip(*perfs)
            plt.plot(xs, 100*np.array(aucs), fmt, label=name)
    plt.title(title, fontdict = {'fontsize' : 20})
    if legend:
        plt.legend(prop={'size':20})
    if mark:
        plt.xlabel('Number Explored')
        plt.ylabel('MSE')
    plt.tight_layout(pad=0.0)
    plt.savefig(save_fname)
    plt.show()
    
def write(tpls, task):
    for arr, cal_type in tpls:
        fname = "../data/%s/%s.pkl" % (task, cal_type)
        with open(fname, "wb") as f:
            pickle.dump(arr, f)
            

import celeba, celeba_private, cocos3, cocos3_10k
dev = torch.device("cpu")

task = "celeba_private"
celeba.root_dir = "../data/celeba"
cocos3.CACHE_DIR = "../data/cocos3"
cocos3_10k.CACHE_DIR = "../data/cocos3_10k"
celeba_private.root_dir = "../data/celeba_private"
celeba_private.celeba_root_dir = "../data/celeba"
celeba_private.private_celeba_root_dir = "../../age-gender-estimation"
calibrated_perfs, noep_calibrated_perfs, uncalibrated_perfs, gt_perfs = [], [], [], []
debug_objs = []
err_ns = [0., 0.3, 0.7]
sts = ["correctedwep"] #, "correctednoep", "raw", "gt"]

def perf_wpool(num_sample, sample_type):
    err1s, err2s, err3s = [], [], []
    for seed in range(3):
        if task == "celeba":
            dataset, data_fitter = celeba.prepare(seed)
        elif task == "celeba_private":
            dataset, data_fitter = celeba_private.prepare(seed)
        elif task == "cocos3":
            dataset, data_fitter = cocos3.prepare(seed)
        elif task == "cocos3_10k":
            dataset, data_fitter = cocos3_10k.prepare(seed)
        explorer = beta_explorer.BetaExplorer(dataset, data_fitter, data_fitter.cache_dir, dev, 
                                              explore_strategy='variance', err_ns=err_ns)
        np.random.seed(seed)
        err_1, err_2, err_3, all_err = explorer.brute_predictor_wpool(num_sample, width=3, sample_type=sample_type)
        err1s.append(err_1)
        err2s.append(err_2)
        err3s.append(err_3)
    return np.mean(err1s), np.mean(err2s), np.mean(err3s)

for num_sample in [500, 1500, 3000]:
    for sti, st in enumerate(sts):
        err_1, err_2, err_3 = perf_wpool(num_sample, st)
        if sti == 0:        
            calibrated_perfs.append((num_sample, err_1, err_2, err_3))
        elif sti == 1:
            noep_calibrated_perfs.append((num_sample, err_1, err_2, err_3))
        elif sti == 2:
            uncalibrated_perfs.append((num_sample, err_1, err_2, err_3))   
        elif sti == 3:
            gt_perfs.append((num_sample, err_1, err_2, err_3))   
    
write([(calibrated_perfs, "calwpool:full")], task)
print (calibrated_perfs)