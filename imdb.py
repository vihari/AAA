import numpy as np
import tqdm
import os
from PIL import Image, ImageDraw
import torch
from torchvision import models, datasets, transforms
import argparse
import sys
import gc
import pickle

from src.utils import train_utils, config, data_utils, misc
from src import dataset, data_fitter, bern_gp_rloss_explorer, beta_gp_rloss_explorer, betaab_gp_rloss_explorer, simple_explorer

dev = config.device

root_dir = "data/imdb"
celeba_root_dir = "data/celeba"
# private_celeba_root_dir = "../age-gender-estimation"
private_celeba_root_dir = "data/service_cache/mf_imdb"
celeba_root = os.path.expanduser("~/datasets")

"""
Same as CelebA but the primary model replaced with a powerful pretrained model.
https://github.com/yu4u/age-gender-estimation
"""

all_attrs = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]
gen_params = {
    'Black_Hair': np.arange(2), 
    'Blond_Hair': np.arange(2), 
    'Brown_Hair': np.arange(2),
    'Smiling': np.arange(2),
    'Male': np.arange(2),
    'Chubby': np.arange(2),
    'Mustache': np.arange(2),
    'No_Beard': np.arange(2),
    'Wearing_Hat': np.arange(2),
    'Blurry': np.arange(2), 
    'Young': np.arange(2),
    'Eyeglasses': np.arange(2),
}
subset_idxs = [all_attrs.index(attr) for attr in gen_params]
    
D = np.meshgrid(*[np.arange(len(gen_params[key])) for key in gen_params])
attr_dim = len(gen_params)
for _ in range(attr_dim):
    D[_] = np.reshape(D[_], -1)
# -1 x attr_dim
D = np.stack(D, axis=1)
print ("Arms shape: ", np.shape(D), D[0], D[-1])


# Common transform for these images
target_resolution = (224, 224)
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(
        target_resolution,
        scale=(0.7, 1.0),
        ratio=(1.0, 1.3333333333333333),
        interpolation=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tf = transforms.Compose([
    transforms.Resize(target_resolution),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def subset_attr_fn(target):
    """
    Helper function that transforms the 40 bit vector in to len(gen_params) bit vector in that order
    """
    assert len(target)==40, "Length of taget is of unexpected size!!"
    assert len(np.shape(target)) == 1
    return np.array([target[idx] for idx in subset_idxs])


class CelebADataset(dataset.Dataset):
    def __init__(self, seed=0):
        self.seed = seed
        np.random.seed(seed)
        self._arms = D
        
        self.primary_task = 4
        celeba_data = datasets.CelebA(root=celeba_root, split='all', target_type='attr', transform=np.array, target_transform=subset_attr_fn, download=False)

        lab_idxs = np.load("%s/lab_idxs.npy" % root_dir)
        unlab_idxs = np.load("%s/unlab_idxs.npy" % root_dir)

        # prepare a reverse lookup index from arm hash to index
        self.arm_hash_to_index = {self.hash_arm(self.arms[arm_index]): arm_index for arm_index in range(len(self.arms))}
        print ("Number of arms: %d over %d instances -- shape: %s" % (len(self.arms), len(celeba_data), self._arms.shape))
        
        def hash_attrs(attrs):
            hashed_arms = [self.hash_arm(subset_attr_fn(_a.numpy())) for _a in attrs]
            arm_indices = [self.arm_hash_to_index[_ha] for _ha in hashed_arms]
            y = [self._arms[arm_index][self.primary_task] for arm_index in arm_indices]
            return np.array(y), np.array(arm_indices)

        class ImgDataset:
            def __init__(self, dataset):
                self.dataset = dataset
            def __getitem__(self, idx):
                return self.dataset[idx][0]
            def __len__(self):
                return len(self.dataset)
        
        with open("%s/subset_lab_idxs_seed=%d.pkl" % (celeba_root_dir, self.seed), "rb") as f:
            self.subset_label_idxs = pickle.load(f)
        l_dataset = torch.utils.data.Subset(celeba_data, lab_idxs)
        l_bitvectors = [celeba_data.attr[idx] for idx in lab_idxs]
        labeled_x = ImgDataset(l_dataset)
        labeled_y, labeled_a = hash_attrs(l_bitvectors)
        self.labeled_data = (np.arange(len(self.subset_label_idxs)), labeled_y[self.subset_label_idxs], labeled_a[self.subset_label_idxs])
        
        ul_dataset = torch.utils.data.Subset(celeba_data, unlab_idxs)
        unlab_x = ImgDataset(ul_dataset)
        ul_bitvectors = [celeba_data.attr[idx] for idx in unlab_idxs]
        unlab_y, unlab_a = hash_attrs(ul_bitvectors)
        print ("Size of available unlabeled data: %d" % len(unlab_x))
        self.U = (np.arange(len(self.subset_label_idxs), len(self.subset_label_idxs) + len(unlab_idxs)), unlab_y, unlab_a)

        self.arm_to_idxs = {}
        for ai, arm_index in enumerate(unlab_a):
            self.arm_to_idxs[arm_index] = self.arm_to_idxs.get(arm_index, []) + [ai]
        
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
        """
        Returns all the available data
        :return: np.arrays of (x, y, arm_indices)
        same interpretation of notation as self.sample
        """
        return self.U
        
    def num_attrs(self):
        """
        :return: number of attributes 
        """
        return np.shape(self.arms)[-1]

    def __len__(self):
        """
        :return: length of available data
        """
        return len(self.U[0])
    
    def hash_arm(self, arm):
        """
        :return: signature string of arm with arm_index
        """
        return "::".join(map(str, arm))
        
    def hash_to_arm_index(self, hashed_arm: str):
        return self.arm_hash_to_index[hashed_arm]

class ModelFromCache(dataset.ModelFromCache):
    def __init__(self, attr, seed):
        """
        :param attr: index in gen_params
        """
        attr_name = list(gen_params.keys())[attr]
        if attr_name == "Male":
            with open("%s/lab_probs.pkl" % private_celeba_root_dir, "rb") as f:
                lab_probs = pickle.load(f)
            with open("%s/unlab_probs.pkl" % private_celeba_root_dir, "rb") as f:
                unlab_probs = pickle.load(f)
            with open("%s/subset_lab_idxs_seed=%d.pkl" % (celeba_root_dir, seed), "rb") as f:
                subset_label_idxs = pickle.load(f)
            lab_probs = np.array(lab_probs)[subset_label_idxs]
            probs = np.concatenate([np.array(lab_probs), np.array(unlab_probs)], axis=0)
            self.logits = probs
        else:
            with open("%s/lab_probs_seed=%d.pkl" % (celeba_root_dir, seed), "rb") as f:
                lab_probs = pickle.load(f)
            with open("%s/unlab_probs.pkl" % celeba_root_dir, "rb") as f:
                unlab_probs = pickle.load(f)
            probs = np.concatenate([lab_probs[attr], unlab_probs[attr]], axis=0)
            self.logits = probs
            
def prepare(seed):
    """
    Prepares data, models and resturns dataset, fitter object for consumption
    """                
    print ("Loading label attribute models...")
    keys = list(gen_params.keys())
    
    print ("Loding dataset...")
    celeba_dataset = CelebADataset(seed)
    print ("Setting models...")
    model_helpers = [ModelFromCache(attr, seed) for attr in range(len(gen_params))]
    print ("Initializing fitter...")
    l_np_x, l_np_y, l_np_a = celeba_dataset.labeled_data
    
    config = data_fitter.Config()
    celeba_data_fitter = data_fitter.Fitter(celeba_dataset, model_helpers, dev, cache_dir=root_dir, config=config)
    celeba_data_fitter.set_primary_task_index(celeba_dataset.primary_task)
    
    # prepare a kernel to obtain features from attr vec
    in_features = len(gen_params)
    # There are only 650 arms active, make sure the model has comparable number of params
    feature_extractor = torch.nn.Sequential(
        torch.nn.Linear(in_features, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 20)
    )
    celeba_data_fitter.set_deep_kernel(feature_extractor, 20)
    
    # all set
    if not os.path.exists(celeba_data_fitter.model_name):
        celeba_data_fitter.fit(use_edge_potentials=True)
    if not os.path.exists(celeba_data_fitter.model_name_no_edge_potential):
        celeba_data_fitter.fit(use_edge_potentials=False)
    
    return celeba_dataset, celeba_data_fitter
    
from src.utils import param_config
if __name__ == '__main__':    
    parser = misc.get_arg_parser()
    
    args = parser.parse_args()
    print ("Check", args.freq_alpha)
    if args.ablation:
        print ("Ablation study...")
    celeba_dataset, celeba_data_fitter = prepare(args.seed)
    accs = [(ai, _acc) for ai, _acc in enumerate(celeba_data_fitter.arm_index_to_acc) if not np.isnan(_acc)]
    print ("Bad arms")
    for ai, _acc in sorted(accs, key=lambda _: _[1])[:10]:
        print (celeba_dataset.arms[ai], _acc, len(celeba_dataset.arm_to_idxs[ai]))
    print ("Good arms")
    for ai, _acc in sorted(accs, key=lambda _: -_[1])[:10]:
        print (celeba_dataset.arms[ai], _acc, len(celeba_dataset.arm_to_idxs[ai]))

    fargs = [celeba_dataset, celeba_data_fitter, celeba_data_fitter.cache_dir, dev]
    fkwargs = {"explore_strategy":args.et, "seed": args.seed, "sample_type": args.sample_type}
    if args.explorer == "bern_gp_rloss":
        fkwargs["width"] = args.width
        if args.ablation:
            bern_gp_rloss_explorer.estimation_ablation(fargs, fkwargs)
        else:
            explorer = bern_gp_rloss_explorer.BernGPExplorer(*fargs, **fkwargs)
    elif args.explorer == "beta_gp_rloss":
        misc.populate_params(fkwargs, args)
        fkwargs["lr"] = 1e-3
        if args.ablation:
            if args.alpha_beta:
                betaab_gp_rloss_explorer.estimation_ablation(fargs, fkwargs)
            else:
                beta_gp_rloss_explorer.estimation_ablation(fargs, fkwargs)
        else:
            celeba_data_fitter.SAMPLE_TOL = 0.1
            explorer = beta_gp_rloss_explorer.BetaGPExplorer(*fargs, **fkwargs)
    elif args.explorer == 'simple':
        if args.ablation:
            simple_explorer.estimation_ablation(fargs, fkwargs)
        else:
            explorer = simple_explorer.SimpleExplorer(*fargs, **fkwargs)
        
    if not args.ablation:
        explorer.explore_and_fit(budget=2000)