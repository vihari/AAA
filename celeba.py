import numpy as np
import tqdm
import os
from PIL import Image, ImageDraw
import torch
from torchvision import models, datasets, transforms
import argparse
import sys
import gc

from src.utils import train_utils, config, data_utils, misc
from src import dataset, data_fitter, bern_gp_rloss_explorer, beta_gp_rloss_explorer, betaab_gp_rloss_explorer, simple_explorer

if config.xla:
    import torch_xla
    import torch_xla.core.xla_model as xm 
    os.environ["XRT_TPU_CONFIG"] = "tpu_worker;0;xxxx:8470"
    dev = xm.xla_device()
else:
    dev = config.device

a = 10
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

root_dir = "data/celeba"
celeba_root = os.path.expanduser("~/datasets")

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

class MultiOutputModel(torch.nn.Module):
    def __init__(self, num_attrs):
        super(MultiOutputModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.feature_extractor = self.resnet_features
        self.sms = torch.nn.ModuleList([torch.nn.Linear(self.model.fc.in_features, 2) for _ in range(num_attrs)])
        
        self.num_attrs = num_attrs
            
    def resnet_features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = [sm(features) for sm in self.sms]
        # batch_size x num_attrs x 2
        logits = torch.stack(logits, dim=1)
        return logits
        
    def loss(self, logits, labels):
        """
        :param logits: batch_size x num_attrs x num_classes size tensor of unscaled preds
        :param labels: batch_size x num_attrs size tensor of ground truth
        returns loss
        """
        assert (logits.shape[1] == self.num_attrs) and (logits.shape[2] == 2), "Unexpected shape of logits: %s" % logits.shape
        assert labels.shape[1] == self.num_attrs, "Unexpected shape of labels: %s" % labels.shape
        assert labels.shape[0] == logits.shape[0]
        batch_size = logits.shape[0]
        logits = torch.reshape(logits, [-1, 2])
        labels = torch.reshape(labels, [-1])
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return torch.mean(loss)*self.num_attrs
    
def evaluate(model, eval_loader, dev):
    model.eval()
    num = 0
    z = np.zeros(model.num_attrs)
    for torch_x, torch_y in tqdm.tqdm(eval_loader):
        if dev is not None:
            torch_x = torch_x.to(dev)
            torch_y = torch_y.to(dev)
        logits = model(torch_x)
        preds = torch.argmax(logits, dim=-1)
        per_attr_corr = (preds == torch_y).type(torch.float32).sum(dim=0)
        z += per_attr_corr.detach().cpu().numpy()
        num += len(torch_x)
    z /= num
    stats = {'acc': np.mean(z), 'per_attr_acc': z}
    return stats
    
def train(train_loader, val_loader, num_attrs, save_name, nepochs=20, dev=None):
    """
    Train helper for a simple classification model
    """
    model = MultiOutputModel(num_attrs)
    if dev is not None:
        model = model.to(dev)
    
    best_val_acc, prev_val_acc = -1, -1
    strikes, tol = 0, 500
    initial_lr = 3e-4
    optimizer = torch.optim.SGD(model.parameters(), initial_lr, momentum=0.9)
    print ([n for n, _ in list(model.named_parameters())])
    ce_loss = torch.nn.CrossEntropyLoss()
    
    for epoch in tqdm.tqdm(range(nepochs)):
        model.train()
        print("Epoch %d start" % epoch, model.sms[0].weight)
        for torch_x, torch_y in tqdm.tqdm(train_loader):
            if dev is not None:
                torch_x = torch_x.to(dev)
                torch_y = torch_y.to(dev)
            
            logits = model(torch_x)
            loss = model.loss(logits, torch_y)
            optimizer.zero_grad()
            loss.backward()
            if dev.type == 'cpu':
                optimizer.step()
            else:
                xm.optimizer_step(optimizer, barrier=True)
            
        print("Epoch %d end" % epoch, model.sms[0].weight)

        print ("Evaluating...")
        val_acc = evaluate(model, val_loader, dev)['acc']
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.cpu().state_dict(), save_name)
            # move it back to the device
            model = model.to(dev)
        sys.stderr.write ("Epoch: %d Val. Acc: %0.4f Best: %0.4f\n" % (epoch, val_acc, best_val_acc))

        if val_acc < prev_val_acc:
            strikes += 1

        if strikes >= tol:
            break
        prev_val_acc = val_acc

    model.load_state_dict(torch.load(save_name))
    model = model.to(dev)
    sys.stderr.write ("Best val. accuracy: %0.4f\n" % (evaluate(model, val_loader, dev)['acc']))
    return model

def train_models():
    print ("Save dir:", root_dir)
    D = arms = np.load("%s/arms.npy" % root_dir)
    print ("Loaded arms arrays with %d entries" % len(arms))

    celeba_data = datasets.CelebA(root=celeba_root, split='all', target_type='attr', transform=tf, target_transform=subset_attr_fn, download=False)

    np.random.seed(0)
    torch.manual_seed(0)
    all_idxs = np.arange(len(celeba_data))
    np.random.shuffle(all_idxs)
    lab_size = 50000
    lab_idxs, unlab_idxs = all_idxs[:lab_size], all_idxs[lab_size:3*lab_size]
    np.save("%s/lab_idxs.npy" % root_dir, lab_idxs)
    np.save("%s/unlab_idxs.npy" % root_dir, unlab_idxs)
    print ("Size of labeled and unlabeled data: %d %d" % (len(lab_idxs), len(unlab_idxs)))
    
    num_attrs = np.shape(arms)[-1]
    
    lab_celeba = torch.utils.data.Subset(celeba_data, lab_idxs)
    train_num, val_num = int(len(lab_celeba)*0.6), int(len(lab_celeba)*0.1)
    train_val_set, test_set = torch.utils.data.random_split(lab_celeba, [train_num + val_num, len(lab_celeba) - train_num - val_num])
    print ("Size of train val set: %d, test set: %d" % (len(train_val_set), len(test_set)))
    train_set, val_set = torch.utils.data.random_split(train_val_set, [train_num, val_num])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128)

    save_name = "%s/attr_model.pb" % root_dir
    model = train(train_loader, val_loader, nepochs=50, num_attrs=len(gen_params), 
                  save_name=save_name, dev=dev)

    test_acc = evaluate(model, test_loader, dev=dev)['per_attr_acc']
    print ("Test accuracies:")
    for ai, attr in enumerate(gen_params.keys()):
        print ("%s: %0.4f" % (attr, test_acc[ai]))


class CelebADataset(dataset.Dataset):
    def __init__(self, seed=0):
        self.seed = seed
        np.random.seed(seed)
        self._arms = D
        
        self.primary_task = 4
        celeba_data = datasets.CelebA(root=celeba_root, split='all', target_type='attr', transform=np.array, target_transform=subset_attr_fn, download=False)

        lab_idxs = np.load("%s/lab_idxs.npy" % root_dir)
        unlab_idxs = np.load("%s/unlab_idxs.npy" % root_dir)

#         bitvectors = [celeba_data.attr[idx] for idx in range(len(celeba_data))]
#         arm_hashes = set()
#         self._arms = []
#         for _a in bitvectors:
#             _hash = self.hash_arm(subset_attr_fn(_a.numpy()))
#             if _hash not in arm_hashes:
#                 self._arms.append(subset_attr_fn(_a.numpy()))
#                 arm_hashes.add(_hash)
#         self._arms = np.array(self._arms)
        # prepare a reverse lookup index from arm hash to index
        self.arm_hash_to_index = {self.hash_arm(self.arms[arm_index]): arm_index for arm_index in range(len(self.arms))}
        print ("Number of arms: %d over %d instances -- shape: %s" % (len(self.arms), len(celeba_data), self._arms.shape))
#         print (self.arm_hash_to_index)
        
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
        
        l_dataset = torch.utils.data.Subset(celeba_data, lab_idxs)
        l_bitvectors = [celeba_data.attr[idx] for idx in lab_idxs]
        labeled_x = ImgDataset(l_dataset)
        labeled_y, labeled_a = hash_attrs(l_bitvectors)
        self.labeled_data = (labeled_x, labeled_y, labeled_a)
        
        ul_dataset = torch.utils.data.Subset(celeba_data, unlab_idxs)
        unlab_x = ImgDataset(ul_dataset)
        ul_bitvectors = [celeba_data.attr[idx] for idx in unlab_idxs]
        unlab_y, unlab_a = hash_attrs(ul_bitvectors)
        self.U = (unlab_x, unlab_y, unlab_a)

        self.arm_to_idxs = {}
        for ai, arm_index in enumerate(unlab_a):
            self.arm_to_idxs[arm_index] = self.arm_to_idxs.get(arm_index, []) + [ai]
        print ("Size of available unlabeled data: %d" % len(unlab_x))
        self.U = (unlab_x, unlab_y, unlab_a)
        
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
        # limit only to 500 random examples
        labeled_x, labeled_y, labeled_a = self.labeled_data
        np.random.seed(self.seed)
        idxs = data_utils.cache(
                lambda: np.random.choice(len(labeled_x), 500), 
                "%s/subset_lab_idxs_seed=%d.pkl" % (root_dir, self.seed),
                use_cache=True
            )
        subset_labeled_x = torch.utils.data.Subset(labeled_x, idxs)

        return (subset_labeled_x, labeled_y[idxs], labeled_a[idxs])
    
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

class SingleOutputModel(dataset.Model):
    def __init__(self, multi_output_model, index, dev, x_transform):
        model = multi_output_model.to(dev)
        model.eval() 
        self._model = model
        def indexed_fn(*args, **kwargs):
            return model(*args, *kwargs)[:, index, :]
        self.model = indexed_fn
        self.transform = x_transform
        self.index = index
        self.dev = dev
                        
class MultiOutputModel2:
    """
    Different from MultiOutputModel in that this can operate on untransformed input 
    """
    def __init__(self, multi_output_model, dev, x_transform):
        self.dev = dev
        self.transform = x_transform
        self.model = multi_output_model
        self.model.eval()
        
    def logit_per_attr(self, np_x, debug=False):
        dataset = data_utils.DatasetNumpy(np_x, np.zeros([len(np_x)]), transform=self.transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128)
        logit_list = []
        if debug: 
            loader = tqdm.tqdm(loader, desc='logits..')
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.dev)
                # batch_size x num_attrs x 2
                logits = self.model(x)
                np_logits = logits.detach().cpu().numpy()
                logit_list.append(np_logits)
                del logits, x
                gc.collect()
        np_logits = np.concatenate(logit_list, axis=0)
        num_attrs = np_logits.shape[1]
        return [np_logits[:, ai, :] for ai in range(num_attrs)]
        
def prepare(seed):
    """
    Prepares data, models and resturns dataset, fitter object for consumption
    """        
    # Train the attribute models if not present
#     if not os.path.exists("%s/attr_model.pb" % root_dir):
#     train_models()
        
    print ("Loading label attribute models...")
    keys = list(gen_params.keys())
    save_name = "%s/attr_model_backup.pb" % root_dir
    model = MultiOutputModel(len(gen_params))
    model = model.to(dev)
    model.load_state_dict(torch.load(save_name))
    print ("Done")
    model = model.to(dev)
        
    print ("Loding dataset...")
    celeba_dataset = CelebADataset(seed)
    print ("Setting models...")
    model_helpers = [SingleOutputModel(model, attr, dev, x_transform=tf) for attr in range(len(gen_params))]
    joint_model = MultiOutputModel2(model, dev, x_transform=tf)
    print ("Initializing fitter...")
    l_np_x, l_np_y, l_np_a = celeba_dataset.labeled_data
    
    celeba_data_fitter = data_fitter.Fitter(celeba_dataset, model_helpers, dev, joint_model=joint_model, cache_dir=root_dir)
    celeba_data_fitter.set_primary_task_index(celeba_dataset.primary_task)
    
    # prepare a kernel to obtain features from attr vec
    in_features = len(gen_params)
    # There are only 400 arms active, make sure the model has comparable number of params
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
        fkwargs["lr"] = 1e-2
        if args.ablation:
            bern_gp_rloss_explorer.estimation_ablation(fargs, fkwargs)
        else:
            explorer = bern_gp_rloss_explorer.BernGPExplorer(*fargs, **fkwargs)
    elif args.explorer == "beta_gp_rloss":
        misc.populate_params(fkwargs, args)
        fkwargs["lr"] = 1e-2
        if args.ablation:
            if args.alpha_beta:
                betaab_gp_rloss_explorer.estimation_ablation(fargs, fkwargs)
            else:
                beta_gp_rloss_explorer.estimation_ablation(fargs, fkwargs)
        else:
            explorer = beta_gp_rloss_explorer.BetaGPExplorer(*fargs, **fkwargs)    
    elif args.explorer == 'simple':
        if args.ablation:
            simple_explorer.estimation_ablation(fargs, fkwargs)
        else:
            explorer = simple_explorer.SimpleExplorer(*fargs, **fkwargs)
        
    if not args.ablation:
        explorer.explore_and_fit(budget=2000)