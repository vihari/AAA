import numpy as np
import os
from PIL import Image
import tqdm
from src.utils import data_utils, misc
from src import dataset
from multiprocessing import Pool
import pickle
import argparse

import torch

from src import data_fitter
from src.utils import train_utils, config, data_utils
from src import dataset, data_fitter, bern_gp_rloss_explorer, beta_gp_rloss_explorer, betaab_gp_rloss_explorer, simple_explorer

ROOT_DIR = os.path.expanduser("~/datasets/cocostuff")
CACHE_DIR = "data/cocos3"
preds_dir = os.path.expanduser("~/repos/deeplab-pytorch/data/features/cocostuff164k/deeplabv2_resnet101_msc")
primary_label_strs = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
scene_label_hierarchy = {
    "water-other": ["sea", "river"],
    "ground-other": ["ground-other", "playingfield", "platform", "railroad", "pavement", "road", "gravel", "mud", "dirt", "snow", "sand", "solid-other", "hill", "mountain", "stone", "rock", "wood", "plant-other", "straw", "moss", "branch", "flower", "leaves", "bush", "tree", "grass"],
    "sky-other": ["sky-other", "clouds"], 
    "structural-other": ["structural-other", "cage", "fence", "railing", "net", "building-other", "house", "roof", "tent", "skyscraper", "bridge"], 
    "furniture-other": ["furniture-other", "stairs", "light", "counter", "mirror-stuff", "cupboard", "shelf", "cabinet", "table", "desk-stuff", "door-stuff", "window-other", "window-blind", "metal", "plastic", "cardboard", "paper", "floor-other", "floor-stone", "floor-marble", "floor-wood", "floor-tile", "carpet", "ceiling-other", "ceiling-tile", "wall-other", "wall-concrete", "wall-stone", "wall-brick", "wall-concrete", "wall-tile", "wall-panel"],
}
# scene_label_strs = ["sea", "fog", "ground-other", "railroad", "road", "snow", "sand", "hill", "wood", "tree", "bush", "grass", "railing", "fence", "bridge", "house", "sky-other", "textile-other", "furniture-other", "window-other", "floor-other", "wall-other", "plastic"]
dev = config.device

gen_params = {
    'animal': np.arange(10), 
    'water': np.arange(2), 
    'ground': np.arange(2), 
    'sky': np.arange(2),
    'building': np.arange(2),
    'furniture': np.arange(2),
}

D = np.meshgrid(*[np.arange(len(gen_params[key])) for key in gen_params])
attr_dim = len(gen_params)
for _ in range(attr_dim):
    D[_] = np.reshape(D[_], -1)
# -1 x attr_dim
D = np.stack(D, axis=1)
print ("Arms shape: ", np.shape(D), D[0], D[-1])


def get_labels():
    labels_strs = []
    # The checkpoint from: https://github.com/kazuto1011/deeplab-pytorch does not use unlabeled label
    # so are the annotations images of cocos.
    with open(os.path.join(CACHE_DIR, "labels.txt")) as f:
        for line in f:
            line = line.strip()
            flds = line.split()
            if flds[1] == "unlabeled":
                continue
            labels_strs.append(flds[1])
    
    primary_label_map = dict([(labels_strs.index(_pl), pi) for pi, _pl in enumerate(primary_label_strs)])
    scene_label_map = dict([(labels_strs.index(_sl), si) for si, _sl in enumerate(scene_label_hierarchy)])
    fine_to_coarse_scene_index = {}
    for k in scene_label_hierarchy:
        for fine_label in scene_label_hierarchy[k]:
            fine_index = labels_strs.index(fine_label)
            coarse_index = labels_strs.index(k)
            fine_to_coarse_scene_index[fine_index] = coarse_index
#     scene_label_map = dict([(sl, si) for si, sl in enumerate(range(92, 183))])
    
    print (fine_to_coarse_scene_index)
    return primary_label_map, scene_label_map, fine_to_coarse_scene_index

def _get_ids():
    train_ids = [fname[:-4] for fname in os.listdir(preds_dir + "/train2017/logit/") if fname.endswith(".npy")]
    val_ids = [fname[:-4] for fname in os.listdir(preds_dir + "/val2017/logit/") if fname.endswith(".npy")]
    return train_ids, val_ids


def _filter_fn(obj):
    fname, _id = obj
    img = Image.open(fname)
    img = np.array(img)
    labels = np.unique(img)
    _pls = [label for label in labels if label in primary_label_map]
    if len(_pls) == 1:
        return (_id, True)
    return (_id, False)


def filter_ids_with_single_object(ids, train=True):
    good_ids = []
    
    fnames = []
    for _id in ids:
        fname = _id + ".png"
        fname = ROOT_DIR + "/dataset/annotations/" + ("train2017" if train else "val2017") + "/" + fname
        fnames.append((fname, _id))

    with Pool(20) as p:
        all_ids = list(tqdm.tqdm(p.imap(_filter_fn, fnames), total=len(ids)))
    good_ids = [_id for _id, _good in all_ids if _good]
    return good_ids
   

def fetch_preds(id_train_pair):
    _id, train = id_train_pair
    primary_labels = np.array([_pl for _pl in primary_label_map])
    scene_labels = np.array([_sl for _sl in scene_label_map])
    
    fldr = "train2017" if train else "val2017"
    pred_fname = preds_dir + "/" + fldr + "/logit/" + _id + ".npy"
    assert os.path.exists(pred_fname)

    logits = np.load(pred_fname)
    preds = np.argmax(logits, axis=0)
    pls = [primary_label_map[_pl] for _pl in np.unique(preds) if _pl in primary_labels]
    random_pick = 0
    # this could happen
    if len(pls) > 1:
        scores = [len(np.where(preds==_pl)[0]) for _pl in pls]
        # retain the label with the highest prevalence
        good_pl_idx = np.argmax(scores)
        pl = pls[good_pl_idx]
    elif len(pls) == 0:
        primary_preds = np.argmax(logits[primary_labels, :, :], axis=0)
        pls = np.unique(primary_preds)
        scores = [len(np.where(primary_preds==_pl)[0]) for _pl in pls]
        good_pl_idx = np.argmax(scores)
        pl = pls[good_pl_idx]
        random_pick = 1
    elif len(pls) == 1:
        pl = pls[0]
    probs = torch.softmax(torch.tensor(logits), dim=0)
    primary_prob_cumsum = torch.mean(probs[primary_labels, :, :], dim=[1, 2]).numpy()

    z = np.zeros(len(scene_label_map))
    scene_probs = np.zeros([len(scene_label_map), 2])
    scene_coarse_preds = [fine_to_coarse_scene_index[_pl] for _pl in preds.flatten() if _pl in fine_to_coarse_scene_index]
    for fine_index, coarse_index in fine_to_coarse_scene_index.items():
        _cl = scene_label_map[coarse_index]
        scene_probs[_cl, 1] += torch.mean(probs[fine_index, :, :]).numpy()
        scene_probs[_cl, 0] += torch.mean(1 - probs[fine_index, :, :]).numpy()
    scene_coarse_labels = np.unique(scene_coarse_preds)
    scores = [len(np.where(scene_coarse_preds==_pl)[0]) for _pl in scene_coarse_labels]
    for si in np.argsort(-np.array(scores)):
        support = scores[si]
        _pl = scene_coarse_labels[si]
        z[scene_label_map[_pl]] = 1
        
    return {"id": _id, "pl": pl, "sl": z, 
            "random_pick": random_pick, 
            "primary_logits": primary_prob_cumsum,
            "scene_logits": scene_probs}


def predictions(ids, train=True):
    all_pl, all_sl = [], []
    logits_pl, logits_sl = [], []
    with Pool(5) as p:
        objs = list(tqdm.tqdm(p.imap(fetch_preds, zip(ids, [train]*len(ids))), total=len(ids)))
    id_to_obj_map =  dict([(obj["id"], obj) for obj in objs])
    num_random = 0
    for _id in ids:
        obj = id_to_obj_map[_id]
        pl, z = obj["pl"], obj["sl"]
        all_pl.append(pl)
        all_sl.append(z)   
        logits_pl.append(obj["primary_logits"])
        logits_sl.append(obj["scene_logits"])
        num_random += obj["random_pick"]
              
    print ("Randomly picked examples: %d" % num_random)
    return all_pl, all_sl, logits_pl, logits_sl


def fetch_gt(id_train_pair):
    _id, train = id_train_pair
    fldr = "train2017" if train else "val2017"
    fname = _id + ".png"
    fname = ROOT_DIR + "/dataset/annotations/" + fldr + "/" + fname
    img = Image.open(fname)
    img = np.array(img)
    gt_labels = np.unique(img)
    pls = [primary_label_map[_pl] for _pl in gt_labels if _pl in primary_label_map]
    assert len(pls) == 1, "Found two primary labels for id: %s, labels: %s" % (_id, pls)
    pl = pls[0]
    gt_labels = [fine_to_coarse_scene_index[_label] for _label in gt_labels if _label in fine_to_coarse_scene_index]
    sls = [scene_label_map[_label] for _label in gt_labels if _label in scene_label_map]
    scores = [len(np.where(img==_label)[0]) for _label in gt_labels if _label in scene_label_map]
    z = np.zeros(len(scene_label_map))
    for si in np.argsort(-np.array(scores)):
        sl = sls[si]
        z[sl] = 1
    return {"id": _id, "pl": pl, "sl": z}


def ground_truth(ids, train=True):
    all_pl, all_sl = [], []
    with Pool(20) as p:
        gts = list(tqdm.tqdm(p.imap(fetch_gt, zip(ids, [train]*len(ids))), total=len(ids)))
    id_to_gt_map = dict([(_gt["id"], _gt) for _gt in gts])
    for _id in ids:
        gt_obj = id_to_gt_map[_id]
        pl, z = gt_obj["pl"], gt_obj["sl"]
        all_pl.append(pl)
        all_sl.append(z)
    return all_pl, all_sl


class COCOSDataset(dataset.Dataset):
    def __init__(self, pred_pls, pred_sls, gt_pls, gt_sls, seed):       
        assert len(pred_pls) == len(pred_sls)
        assert len(gt_pls) == len(gt_sls)
        assert len(pred_pls) == len(gt_pls)
        
        self.seed = seed
        LABELED_DATA_SIZE = 500
        self._arms = D
        #a small subset of labeled
        np.random.seed(0)
        idxs = np.random.permutation(np.arange(len(pred_pls)))
        all_labeled_idxs, unlabeled_idxs = idxs[:3*LABELED_DATA_SIZE], idxs[3*LABELED_DATA_SIZE:]
        np.random.seed(self.seed)
        labeled_idxs = np.random.choice(all_labeled_idxs, LABELED_DATA_SIZE)
        
        self.arm_hash_to_index = {}
        self.arm_to_idxs = {}
        seen_hashes = set()
        # Make arms from what is seen
        for arm in self._arms:
            arm_hash = self.hash_arm(arm)
            if arm_hash not in seen_hashes:
                self.arm_hash_to_index[arm_hash] = len(seen_hashes)
            seen_hashes.add(arm_hash)
                        
        assert len(np.shape(self._arms)) == 2, "Unexpected shape of arms: %s" % np.shape(self._arms)
        
        arm_indices = []
        for idx in range(len(pred_pls)):
            arm = np.concatenate([[gt_pls[idx]], gt_sls[idx]]).astype(np.int32)
            arm_hash = self.hash_arm(arm)
            arm_index = self.arm_hash_to_index[arm_hash]
            arm_indices.append(arm_index)
        arm_indices = np.array(arm_indices)
        
        # this should only keep track of unlabeled indices
        for ui, idx in enumerate(unlabeled_idxs):
            arm = np.concatenate([[gt_pls[idx]], gt_sls[idx]]).astype(np.int32)
            arm_hash = self.hash_arm(arm)
            arm_index = self.arm_hash_to_index[arm_hash]
            x_indices = self.arm_to_idxs.get(arm_index, [])
            x_indices.append(ui)            
            self.arm_to_idxs[arm_index] = x_indices
        
        num_empty = 0
        for ai in range(len(self.arms)):
            if ai not in self.arm_to_idxs:
                num_empty += 1
        print ("Found %d/%d empty arms" % (num_empty, len(self.arms)))
            
        print ("Found %d unique arms and %0.2f average number of examples per arm" % (len(self._arms), np.mean([len(self.arm_to_idxs.get(ai, [])) for ai in range(len(self._arms))])))
        self.labeled_data = (labeled_idxs, gt_pls[labeled_idxs], arm_indices[labeled_idxs])
        self.U = (unlabeled_idxs, gt_pls[unlabeled_idxs], arm_indices[unlabeled_idxs])
        
        assert len(self.arm_hash_to_index) == len(self._arms)

    @property
    def arms(self):
        return self._arms
    
    @property
    def num_arms(self):
        return len(self.arms)
        
    def sample(self, num_sample):
        idxs = np.random.choice(len(self), num_sample)
        x, y, arm_ids = self.U[0][idxs], self.U[1][idxs], self.U[2][idxs]
        return x, y, arm_ids
        
    def sample_arm(self, arm_index, num_sample):
        """
        Sample randomly from arm (integer index) 
        :return: np.array of x, y 
        """
        # allowing repeats
        idxs = np.random.choice(self.arm_to_idxs[arm_index], num_sample)
        x, y, arm_ids = self.U[0][idxs], self.U[1][idxs], self.U[2][idxs]
        return x, y
        
    def full_labeled_data(self):
        return self.labeled_data
    
    def full_data_arm(self, arm_index):
        idxs = self.arm_to_idxs[arm_index]
        x, y = self.U[0][idxs], self.U[1][idxs]
        return x, y
    
    def full_data(self):
        return self.U
        
    def num_attrs(self):
        return np.shape(self.arms)[-1]

    def __len__(self):
        return len(self.U[0])
    
    @staticmethod
    def hash_arm(arm):
        return "::".join(map(str, arm))
        
    def hash_to_arm_index(self, hashed_arm: str):
        # this can happen here since self.arms do not span the universe 
        if hashed_arm not in self.arm_hash_to_index:
            return None
        return self.arm_hash_to_index[hashed_arm]

    
class JointModelFromCache():
    def __init__(self, logits_per_attr):
        """
        Logits should be of shape: [len(full_data) x num_labels_for_this_attr]_{num_attr}
        """
        self.logits = logits_per_attr
        
    def logit_per_attr(self, np_x, debug=False):
        return [self.logits[ai][np_x] for ai in range(len(self.logits))]


def check(gt_pls, gt_sls):
    arm_hash_to_index = {}
    arms = []
    for si in range(len(gt_pls)):
        arm = np.concatenate([[gt_pls[si]], gt_sls[si]]).astype(np.int32)
        arm_hash = COCOSDataset.hash_arm(arm)

        if arm_hash not in arm_hash_to_index:
            arms.append(arm)
        x_indices = arm_hash_to_index.get(arm_hash, [])
        x_indices.append(si)
        arm_hash_to_index[arm_hash] = x_indices
        
    print ("Found %d unique arms and %0.2f average number of examples per arm" % (len(arm_hash_to_index), np.mean([len(arm_hash_to_index[_h]) for _h in arm_hash_to_index])))
    

def _one_hot(preds, depth, on=1, off=0):
    z = np.ones([len(preds), depth])*off
    z[np.arange(len(preds)), preds.astype(np.int64)] = on
    return z


def evaluate(preds, gt):
    corr, num = {}, {}
    for i in range(len(preds)):
        _corr = (preds[i] == gt[i])
        _l = gt[i]
        corr[_l] = corr.get(_l, 0) +_corr
        num[_l] = num.get(_l, 0) + 1

    for k in corr:
        print ("Key: %s Acc: %0.4f num: %d" % (str(k), corr[k]/num[k], num[k]))


def evaluate_sls(pred_sls, gt_sls):
    num_sls = len(pred_sls[0])
    corr = np.zeros(num_sls)
    for i in range(len(pred_sls)):
        corr += (pred_sls[i] == gt_sls[i]).astype(np.float32)

    print ("Per scene label acc:")
    print (corr/len(pred_sls))


def prepare(seed):
    gt_pls, gt_sls, (pred_pls, plogits), (pred_sls, slogits) = parse_preds_gts()
    check(gt_pls, gt_sls)
    # num_examples x num_labels
    plogits = np.array(plogits)
    # num_examples x num_attrs x 2
    slogits = np.array(slogits)
    print ("Shape of p: %s, scene: %s" % (plogits.shape, slogits.shape))
        
    cocos_dataset = COCOSDataset(pred_pls, pred_sls, gt_pls, gt_sls, seed)
    num_sl_attrs = pred_sls.shape[-1]
    
    models = [dataset.ModelFromCache(_one_hot(pred_pls, len(primary_label_map), on=1, off=-1))] + [dataset.ModelFromCache(_one_hot(pred_sls[:, si], 2, on=1, off=-1)) for si in range(num_sl_attrs)]
    joint_model = JointModelFromCache([_one_hot(pred_pls, len(primary_label_map), on=1, off=-1)] + [_one_hot(pred_sls[:, si], 2, on=1, off=-1) for si in range(num_sl_attrs)])

#     models = [dataset.ModelFromCache(plogits)] + [dataset.ModelFromCache(slogits[:, si, :]) for si in range(num_sl_attrs)]
#     joint_model = JointModelFromCache([plogits] + [slogits[:, si, :] for si in range(num_sl_attrs)])
    
    config = data_fitter.Config()
    config.CALIBRATION_TOL = 1e-3
    config.CALIBRATION_TOPK = 5
    
    cocos_fitter = data_fitter.Fitter(cocos_dataset, models=models, device=dev, cache_dir=CACHE_DIR, joint_model=joint_model, config=config)
    cocos_fitter.set_primary_task_index(0)
    
    in_features = 1 + gt_sls.shape[-1]
    kernel_embedding_model = torch.nn.Sequential(
        torch.nn.Linear(in_features, 10)
    )
    cocos_fitter.set_deep_kernel(kernel_embedding_model, 10)
    
    if not os.path.exists(cocos_fitter.model_name):
        cocos_fitter.fit(use_edge_potentials=True)
    if not os.path.exists(cocos_fitter.model_name_no_edge_potential):
        cocos_fitter.fit(use_edge_potentials=False)
        
    return cocos_dataset, cocos_fitter
   
def parse_preds_gts():
    train_ids = data_utils.cache(
        lambda: filter_ids_with_single_object(train_ids, train=True),
        CACHE_DIR + "/train_ids.pkl"
    )
    print ("Found %d good ones in train" % len(train_ids))
    
    val_ids = data_utils.cache(
        lambda: filter_ids_with_single_object(val_ids, train=False),
        CACHE_DIR + "/val_ids.pkl"
    )
    print ("Found %d good ones in val" % len(val_ids))
    
    train_pls, train_sls, train_plogits, train_slogits = data_utils.cache(
        lambda: predictions(train_ids, train=True),
        CACHE_DIR + "/train_preds.pkl"
    )
    val_pls, val_sls, val_plogits, val_slogits = data_utils.cache(
        lambda: predictions(val_ids, train=False),
        CACHE_DIR + "/val_preds.pkl"
    )
    pred_pls, pred_sls = train_pls + val_pls, train_sls + val_sls
    plogits, slogits = train_plogits + val_plogits, train_slogits + val_slogits
    
    train_pls2, train_sls2 = data_utils.cache(
        lambda: ground_truth(train_ids, train=True),
        CACHE_DIR + "/train_gt.pkl"
    )
    val_pls2, val_sls2 = data_utils.cache(
        lambda: ground_truth(val_ids, train=False),
        CACHE_DIR + "/val_gt.pkl"
    )
    gt_pls, gt_sls = train_pls2 + val_pls2, train_sls2 + val_sls2
    
#     evaluate(pred_pls, gt_pls)
    evaluate_sls(pred_sls, gt_sls)
    return np.array(gt_pls), np.array(gt_sls), (np.array(pred_pls), plogits), (np.array(pred_sls), slogits)


primary_label_map, scene_label_map, fine_to_coarse_scene_index = get_labels()
if __name__ == '__main__':
    parser = misc.get_arg_parser()
    
    args = parser.parse_args()
    
    cocos_dataset, cocos_data_fitter = prepare(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    _args = [cocos_dataset, cocos_data_fitter, cocos_data_fitter.cache_dir, dev]
    _kwargs = {'explore_strategy': args.et, 'seed': args.seed, "sample_type": args.sample_type}
    if args.explorer == "bern_gp_rloss":
        _kwargs['width'] = args.width
        if args.ablation:
            bern_gp_rloss_explorer.estimation_ablation(_args, _kwargs)
        else:
            explorer = bern_gp_rloss_explorer.BernGPExplorer(*_args, **_kwargs)
    elif args.explorer == 'beta_gp_rloss':
        misc.populate_params(_kwargs, args)
        if args.ablation:
            if args.alpha_beta:
                betaab_gp_rloss_explorer.estimation_ablation(_args, _kwargs)
            else:
                beta_gp_rloss_explorer.estimation_ablation(_args, _kwargs)
        else:
            explorer = beta_gp_rloss_explorer.BetaGPExplorer(*_args, **_kwargs)
    elif args.explorer == 'simple':
        if args.ablation:
            simple_explorer.estimation_ablation(_args, _kwargs)
        else:
            explorer = simple_explorer.SimpleExplorer(*_args, **_kwargs)
        
    if not args.ablation:
        explorer.explore_and_fit(budget=2000)