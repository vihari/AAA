from abc import ABC
import torch
import tqdm
import numpy as np
import sys, os
import pickle
import collections
import copy

from . import dataset
from .utils import data_utils, train_utils, config

if config.xla:
    import torch_xla.core.xla_model as xm

class Kernel(torch.nn.Module):
    def __init__(self, embedding_model: torch.nn.Module, num_features: int, num_attrs: int, arms: np.ndarray, device):
        """
        :embedding_model: A pre-defined and instantiated embedding model that is used by the kernel
        :num_features: Dimension of the embdding of the embedding model
        :num_attrs: Number of attributes
        :arms: full array of arms (attribute combinations)
        """
        super(Kernel, self).__init__()
        self.embedding_model = embedding_model
        self.emb_size = num_features
        self.model = torch.nn.Sequential(
            self.embedding_model,
            torch.nn.ReLU(),
            torch.nn.Linear(num_features, 1),
        )
        self.temp_param = torch.nn.Parameter(data=torch.zeros([num_attrs]), requires_grad=True)
        self.arms = arms
        self.dev = device
        
    def forward(self, attr_vec: torch.Tensor):
        """
        Gives a score for a given attr_vec, akin to Pr(a)
        TODO: batching?
        """
        return self.model(attr_vec) 
    
    def potential2(self, node_potentials: [np.ndarray], preds: np.ndarray, use_edge_potentials=True, num_sample=30) -> [torch.Tensor]:
        """
        Just like in self.potential, but prob is normalized over all attribute combinations 
        :return:s probability  
        """
        if not hasattr(self, "edge_scores"):
            self.edge_scores = self.model(torch.tensor(self.arms).type(torch.float)).squeeze()
        batch_size = 8
        total_size = len(node_potentials[0])
        top_k = num_sample
        num_attrs = len(node_potentials)
        std_node_potentials = []
                
        best_arms, best_scores = torch.zeros([total_size, top_k], dtype=torch.int32), torch.zeros([total_size, top_k])
        num_steps = int(total_size/batch_size) + 1
        for step in range(num_steps):
            left, right = step*batch_size, min((step+1)*batch_size, total_size)
            node_scores = torch.zeros([right - left, len(self.arms)])
            for ai in range(num_attrs):
                np_this_attr = torch.tensor(node_potentials[ai][left:right])
                node_scores += np_this_attr[:, self.arms[:, ai]] * torch.exp(self.temp_param[ai])            
#             node_scores = node_scores - torch.unsqueeze(torch.logsumexp(node_scores, dim=-1), dim=-1)
            _scores = torch.zeros_like(node_scores)
            _scores += node_scores
            if use_edge_potentials:
                edge_scores = torch.unsqueeze(self.edge_scores, dim=0)
#                 edge_scores = edge_scores - torch.unsqueeze(torch.logsumexp(edge_scores, dim=-1), dim=-1)
                _scores += edge_scores
            top_values, top_indices = torch.topk(_scores, k=top_k)
            best_arms[left:right] = top_indices.type(torch.int32)
            best_scores[left:right] = top_values

        print ("Best scores:")
        print (best_arms[-5:], best_scores[-5:])
        return best_arms, best_scores
    
    def potential(self, node_potentials: [np.ndarray], preds: np.ndarray, use_edge_potentials=True, add=True) -> [torch.Tensor]:
        """
        Combines node potentials with edge potentials and returns array of tensors with potential values.
        The assumption Pr(a_{-j,-k}=a*,a_j\neq a*_j, a_k\neq a^*_k|x) is much less than likelihood of only one misalign, where a^*  is the true attribute.
        :param node_potentials: unnormalized score that models Pr(a_{ij}|x_i) [[-1, size of kth attr]]_{num_attrs}
        :param preds: Predicted attribute value [-1, num_attrs]
        :param use_edge_potentials: When set to False, only node potentials i.e. attribute temperature is trained
        Returns len(node_potentials) length array of [len(preds) x size of kth attr]
        """
        num_attrs = len(node_potentials)
        batch_size = len(node_potentials[0])
        potentials = []
        std_node_potentials = node_potentials
        for k in range(num_attrs):
            new_attr_vecs = []
            attr_k_size = np.shape(node_potentials[k])[-1]
            for v in range(attr_k_size):
                _attr_vecs = preds.copy()
                _attr_vecs[:, k] = v
                new_attr_vecs.append(_attr_vecs) 
            # (batch_size*attr_k_size) x num_attrs
            new_attr_vecs = np.concatenate(new_attr_vecs, axis=0)
            # batch_size*attr_k_size
            edge_scores = torch.squeeze(self.forward(torch.Tensor(new_attr_vecs).to(self.dev)))
            
            idxs = np.reshape([np.arange(batch_size)]*attr_k_size, [batch_size*attr_k_size])

            node_scores = []
            for k1 in range(num_attrs):
                node_scores.append(torch.Tensor(std_node_potentials[k1][idxs, new_attr_vecs[:, k1]]))  
            # batch_size*attr_k_size
            node_scores = torch.stack([node_scores[_]*torch.exp(self.temp_param[_]) for _ in range(num_attrs)], dim=0).sum(dim=0)
            if use_edge_potentials:
                potential_per_attr = node_scores + edge_scores
            else: 
                potential_per_attr = node_scores
            potential_per_attr = torch.reshape(potential_per_attr, [attr_k_size, batch_size]).transpose(1, 0)
            potentials.append(potential_per_attr)
        
        return potentials

class Config:
    """
    Config parameters related to ablation etc. 
    """
    def __init__(self):
        self.CALIBRATION_TOL = 1e-2
        self.CALIBRATION_TOPK = 10
    
    
class Fitter(ABC):
    """
    Probabilistic Edge Potential Fitter on attribute predictions
    After constructing the object, make sure to call 
    set_primary_task_index
    set_deep_kernel
    """
    def __init__(self, dataset: dataset.Dataset, models: [dataset.Model], device: torch.device, cache_dir: str=None, joint_model: dataset.Model=None, config: Config=None):
        """
        :param dataset: Dataset
        :param models: per-attribute predictor
        :param cache_dir: The directory which would be used to keep cached files
        :param joint_model: If all the attributes are modeled jointly, then it is more efficient to do one forward pass for prediction.
        """
        self.dataset = dataset
        self.models = models
        self.config = config
        if config is None:
            self.config = Config()
        
        if self.models is not None:
            assert len(self.models) == self.dataset.num_attrs(), "The number of models argument should be the same size as number of attributes. Expected: %d, found: %d" % (self.dataset.num_attrs(), len(self.models))
        self.dev = device
        self.cache_dir = cache_dir
        self.model_name = "%s/kernel.pb" % cache_dir
        self.model_name_no_edge_potential = "%s/kernel_no_edge_potential.pb" % cache_dir
        self.corrected_dataset = None
        # The index of primary task in the list of models
        self.primary_task_index = 2
        self.joint_model = joint_model
        np.random.seed(0)
        assert (self.models is not None) or (self.joint_model is not None), "The model should be fed either through models or joint model"

        # Any example to arm affiliation score that is smaller than this are pruned. 
        self.SAMPLE_TOL = 0
        
        self.setup()
        
    def setup(self):
        l_np_x, _, _ = self.dataset.full_labeled_data()
        if self.joint_model is None:
            l_node_potentials = data_utils.cache(
                lambda: [model.probs(l_np_x, debug=True) for model in self.models], 
                "%s/lab_probs_seed=%d.pkl" % (self.cache_dir, self.dataset.seed),
                use_cache=self.cache_dir is not None
            )
        else:
            sys.stderr.write ("Using joint model.\n")
            l_node_potentials = data_utils.cache(
                lambda: self.joint_model.logit_per_attr(l_np_x, debug=True), 
                "%s/lab_probs_seed=%d.pkl" % (self.cache_dir, self.dataset.seed),
                use_cache=self.cache_dir is not None
            )
    
    def set_primary_task_index(self, index: int):
        """
        Set index of primary task in the self.models list
        """
        self.primary_task_index = index
        self.primary_model = self.models[index]
        # set off to compute acc per arm once the primary model is set
        self.fit_acc_per_arm()
        
    def set_deep_kernel(self, deep_kernel: torch.nn.modules.container.Sequential, num_features: int):
        """
        Set kernel embedding model that produces an embedding for the attribute vector 
        deep_kernel is a torch Sequential model
        """
        self.kernel_model = Kernel(deep_kernel, num_features, num_attrs=self.dataset.num_attrs(), arms=self.dataset.arms, device=self.dev)
        # copy::deepcopy does not allow shared memory 
        self.kernel_model_no_edge_potential = Kernel(copy.deepcopy(deep_kernel), num_features, num_attrs=self.dataset.num_attrs(), arms=self.dataset.arms, device=self.dev)
        if os.path.exists(self.model_name):
            self.kernel_model.load_state_dict(torch.load(self.model_name))
            sys.stderr.write("Read model from %s\n" % self.model_name)
            sys.stderr.write("Temp: %s\n" % torch.exp(self.kernel_model.temp_param.detach()))
        if os.path.exists(self.model_name_no_edge_potential):
            self.kernel_model_no_edge_potential.load_state_dict(torch.load(self.model_name_no_edge_potential))
            sys.stderr.write("Read model from %s\n" % self.model_name_no_edge_potential)
            sys.stderr.write("Temp: %s\n" % torch.exp(self.kernel_model_no_edge_potential.temp_param.detach()))
        self.kernel_model = self.kernel_model.to(self.dev)
        self.kernel_model_no_edge_potential = self.kernel_model_no_edge_potential.to(self.dev)

    def _populate_corrected_dataset(self):
        if self.corrected_dataset is not None:
            return
        save_name = self.cache_dir + "/corrected_data.pkl"
        if os.path.exists(save_name):
            with open(save_name, "rb") as f:
                self.corrected_dataset = pickle.load(f)
                x, _ = self.corrected_dataset["data"]
                # keep track of points already labeled
                self.availability_array = np.ones(len(x), dtype=np.int32)
                return 
        
        print ("Preparing predicted labels and attributes...")
        num_sample_per_example = self.config.CALIBRATION_TOPK
        tol_prob = self.config.CALIBRATION_TOL
        print ("Calibration topk: %d calibration tol prob: %f" % (num_sample_per_example, tol_prob))
        x, y, _a = self.dataset.full_data()
        with open("%s/unlab_probs.pkl" % self.cache_dir, "rb") as f:
            node_potentials = pickle.load(f)
        preds = [np.argmax(_np, axis=1) for _np in node_potentials]
        preds = np.stack(preds, axis=1)
                
        a = preds
        # num_sample_per_example x len(x) x attr_dim
        sampled_arms, sampled_probs = self.sample_attribute4(preds, node_potentials, num_sample_per_example, use_edge_potentials=True)
        noep_sampled_arms, noep_sampled_probs = self.sample_attribute4(preds, node_potentials, num_sample_per_example, use_edge_potentials=False)
        # convert them to arm indices
        x_index_to_arm_indices = [[] for _ in range(len(preds))]
        arm_idx_to_idxs = [[] for _ in range(self.dataset.num_arms)]
        
        # no edge potentials -- only temperature
        noep_x_index_to_arm_indices = [[] for _ in range(len(preds))]
        noep_arm_idx_to_idxs = [[] for _ in range(self.dataset.num_arms)]

        # with the original predicted attr vector without sampling
        unsampled_x_index_to_arm_indices = [[] for _ in range(len(preds))]
        unsampled_arm_idx_to_idxs = [[] for _ in range(self.dataset.num_arms)]
        
        for xi in tqdm.tqdm(range(len(x)), desc="(Un)Hashing..."):
            this_index_to_arm_indices = []
            cum_prob = 0
            for sample_index in range(num_sample_per_example):
                ai, prob = int(sampled_arms[xi, sample_index]), float(sampled_probs[xi, sample_index])
                cum_prob += prob
                if (prob < tol_prob) and (sample_index > 0):
                    break
            
            for sample_index in range(num_sample_per_example):
                ai, prob = int(sampled_arms[xi, sample_index]), float(sampled_probs[xi, sample_index])
                # in sorted order
                if (prob < tol_prob) and (sample_index > 0):
                    break
                prob /= cum_prob
                arm_idx_to_idxs[ai] += [(xi, prob)]
                this_index_to_arm_indices.append((ai, prob))
            x_index_to_arm_indices[xi] = this_index_to_arm_indices
            
            this_index_to_arm_indices = []
            cum_prob = 0
            for sample_index in range(num_sample_per_example):
                ai, prob = int(sampled_arms[xi, sample_index]), float(sampled_probs[xi, sample_index])
                cum_prob += prob
                if (prob < tol_prob) and (sample_index > 0):
                    break

            for sample_index in range(num_sample_per_example):
                ai, prob = int(noep_sampled_arms[xi, sample_index]), float(noep_sampled_probs[xi, sample_index])
                if prob < tol_prob:
                    break
                prob /= cum_prob
                noep_arm_idx_to_idxs[ai] += [(xi, prob)]
                this_index_to_arm_indices.append((ai, prob))
            noep_x_index_to_arm_indices[xi] = this_index_to_arm_indices
            
            pred_ai = self.dataset.hash_to_arm_index(self.dataset.hash_arm(a[xi]))
            if pred_ai is None:
                # just use the best estimate from calibrated model
                pred_ai = x_index_to_arm_indices[xi][0][0]
            unsampled_x_index_to_arm_indices[xi] = [(pred_ai, 1)]
            unsampled_arm_idx_to_idxs[ai] += [(xi, 1)]
            
        sys.stderr.write("Stats for corrected dataset\n-----------------\n")
        sys.stderr.write ("Average length of mapping from x to arms %f\n" % np.mean([len(_) for _ in x_index_to_arm_indices]))
        sys.stderr.write ("Average length of mapping from arms to x %f\n" % np.mean([len(_) for _ in arm_idx_to_idxs]))
        sys.stderr.write("Stats for (No Edge potential) corrected dataset\n-----------------\n")
        sys.stderr.write ("Average length of mapping from x to arms %f\n" % np.mean([len(_) for _ in noep_x_index_to_arm_indices]))
        sys.stderr.write ("Average length of mapping from arms to x %f\n" % np.mean([len(_) for _ in noep_arm_idx_to_idxs]))
        sys.stderr.write("Stats for original dataset\n-----------------\n")
        sys.stderr.write ("Average length of mapping from x to arms %f\n" % np.mean([len(_) for _ in unsampled_x_index_to_arm_indices]))
        sys.stderr.write ("Average length of mapping from arms to x %f\n" % np.mean([len(_) for _ in unsampled_arm_idx_to_idxs]))
        sys.stderr.write("%s\n" % x_index_to_arm_indices[0])
        
        # x may sometimes be a dataset that cannot be loaded in to memory
        # x is not used as long as we are using primary_model_evaluate_cached instead of evaluate
        x = np.arange(len(x))
        print ("%d %d\n%d %d\n%d %d\n%d %d" % (len(np.array(x)), len(np.array(y)), len(arm_idx_to_idxs), len(x_index_to_arm_indices), len(noep_arm_idx_to_idxs), len(noep_x_index_to_arm_indices), len(unsampled_arm_idx_to_idxs), len(unsampled_x_index_to_arm_indices)))
        self.corrected_dataset = {
            "data": (np.array(x), np.array(y)), 
            "arm_index_to_indices": arm_idx_to_idxs, 
            "x_index_to_arm_indices": x_index_to_arm_indices,
            "noep_arm_index_to_indices": noep_arm_idx_to_idxs, 
            "noep_x_index_to_arm_indices": noep_x_index_to_arm_indices,
            "unsampled_arm_index_to_indices": unsampled_arm_idx_to_idxs,
            "unsampled_x_index_to_arm_indices": unsampled_x_index_to_arm_indices,
        }
        sys.stderr.write("Initiating write...\n")
        with open(save_name, "wb") as f:
            sys.stderr.write("Writing to %s\n" % save_name)
            pickle.dump(self.corrected_dataset, f)
            sys.stderr.write("Done\n")
            
        # keep track of points already labeled
        self.availability_array = np.ones(len(x), dtype=np.int32)

    def primary_model_evaluate_cached(self, idxs):
        """
        Evaluates the accuracy on the examples indexed in to unlabeled data
        returns accuracy of primary model on the requested indices in unlabeled data
        """
        if not hasattr(self, "unlab_corrs"):
            fname = "%s/unlab_corrs.pkl" % self.cache_dir
            if not os.path.exists(fname):
                sys.stderr.write("%s file name is not found!! This is unexpected.\n" % fname)
            with open(fname, "rb") as f:
                self.unlab_corrs = pickle.load(f)
        
        return {"correct": self.unlab_corrs[idxs]}
      
    def warm_start_counts(self, counts0, counts1, x_to_a=None, x_corrs=None):
        """
        Uses all the available data used for calibration to initialize the counts
        """
        l_np_x, l_np_y, l_np_arm_ids = self.dataset.full_labeled_data()
        print ("Using seed: %d for warm start" % self.dataset.seed)
        fname = "%s/lab_probs_seed=%d.pkl" % (self.cache_dir, self.dataset.seed)
        with open(fname, "rb") as f:
            l_probs = pickle.load(f)
        preds = np.argmax(l_probs[self.primary_task_index], axis=-1)
        corrs = (l_np_y == preds).astype(np.float32)
        for corr, arm_idx in zip(corrs, l_np_arm_ids):
            counts0[arm_idx] += (1 - corr)
            counts1[arm_idx] += corr
            if x_to_a is not None:
                x_to_a.append([(arm_idx, 1.)])
            if x_corrs is not None:
                x_corrs.append(corr)
                        
    def get_index_mappings(self, sample_type):
        if self.corrected_dataset is None:
            self._populate_corrected_dataset()
        
        if sample_type == "correctedwep":
            x_index_to_arm_indices = self.corrected_dataset["x_index_to_arm_indices"]
            arm_index_to_indices = self.corrected_dataset["arm_index_to_indices"]
        elif sample_type == "correctednoep":
            x_index_to_arm_indices = self.corrected_dataset["noep_x_index_to_arm_indices"]
            arm_index_to_indices = self.corrected_dataset["noep_arm_index_to_indices"]
        elif sample_type == "raw":
            x_index_to_arm_indices = self.corrected_dataset["unsampled_x_index_to_arm_indices"]
            arm_index_to_indices = self.corrected_dataset["unsampled_arm_index_to_indices"]
        elif sample_type == "gt":
            if "gt_x_index_to_arm_indices" in self.corrected_dataset:
                x_index_to_arm_indices = self.corrected_dataset["gt_x_index_to_arm_indices"]
                arm_index_to_indices = self.corrected_dataset["gt_arm_index_to_indices"]
            else:
                x, y, _a = self.dataset.full_data()
                arm_index_to_indices = [[] for _ in range(self.dataset.num_arms)]
                x_index_to_arm_indices = []
                for xi, ai in zip(range(len(x)), _a): 
                    x_index_to_arm_indices.append([(ai, 1)])
                    arm_index_to_indices[ai].append((xi, 1))
                    
                self.corrected_dataset["gt_x_index_to_arm_indices"] = x_index_to_arm_indices
                self.corrected_dataset["gt_arm_index_to_indices"] = arm_index_to_indices
                
        return x_index_to_arm_indices, arm_index_to_indices
    
    def sample(self, num_sample, counts0, counts1, explored=None, sample_type="correctedwep", replace=False):
        """
        Sample arms and observe the counts to reflect in counts0 and counts1
        Recognised sample_type are "correctedwep", "correctednoep", "raw"
        """
        x_index_to_arm_indices, arm_index_to_indices = self.get_index_mappings(sample_type)
        
        full_x, full_y = self.corrected_dataset["data"]
        if replace:
            available_indices = np.arange(len(self.availability_array))
        else:
            available_indices = np.where(self.availability_array==1)[0]
        
        x_idxs = np.random.choice(available_indices, num_sample, replace=False)
        # TODO: the case when no available indices is unhandled since that is rare
        if not replace:
            assert len(available_indices) >= num_sample, "Number of available indices %d is smaller than requested samples: %d" % (len(available_indices), num_sample)
        x, y = full_x[x_idxs], full_y[x_idxs]
        corrs = self.primary_model_evaluate_cached(x_idxs)["correct"]
#         corrs = self.primary_model.evaluate(x, y, status=False)["correct"]

        for x_idx, c1 in zip(x_idxs, corrs):
            arm_and_wts = x_index_to_arm_indices[x_idx]
            if explored is not None:
                explored[arm_and_wts[0][0]] = 1
            for ai, (arm_idx, wt) in enumerate(arm_and_wts):
                if (ai==0) or (wt > self.SAMPLE_TOL):
                    counts0[arm_idx] += wt*(1 - c1)
                    counts1[arm_idx] += wt*c1
            self.availability_array[x_idx] = 0
                    
    def sample_arm(self, arm_index, num_sample, counts0, counts1, sample_type="correctedwep", x_to_a=None, x_corrs=None):
        """
        returns False when there are no examples available indexed for this arm index or if there are no new samples available for tis index 
        """        
        x_index_to_arm_indices, arm_index_to_indices = self.get_index_mappings(sample_type=sample_type)

        full_x, full_y = self.corrected_dataset["data"]
        if len(arm_index_to_indices[arm_index]) == 0:
            return False
        
        this_idxs = list(map(lambda _: _[0], arm_index_to_indices[arm_index]))
        unsampled_idxs = np.where(self.availability_array==1)[0]
        x_index_to_score = dict([(x_idx, wt) for x_idx, wt in arm_index_to_indices[arm_index]])
        allowed_idxs = np.intersect1d(unsampled_idxs, this_idxs, assume_unique=True)
        if len(allowed_idxs) < num_sample:
            return False
        sorted_allowed_idxs = sorted(allowed_idxs, key=lambda x_idx: x_index_to_score[x_idx], reverse=True)
#         if x_index_to_score[sorted_allowed_idxs[0]] < thresh:
#             return False
        
        sampled_idxs = sorted_allowed_idxs[:num_sample]
        x, y = full_x[sampled_idxs], full_y[sampled_idxs]
        corrs = self.primary_model_evaluate_cached(sampled_idxs)["correct"]
#         corrs = self.primary_model.evaluate(x, y, status=False)["correct"]
        for x_idx, c1 in zip(sampled_idxs, corrs):
            arm_and_wts = x_index_to_arm_indices[x_idx]
            for arm_idx, wt in arm_and_wts:
                if wt > self.SAMPLE_TOL:
                    counts0[arm_idx] += wt*(1 - c1)
                    counts1[arm_idx] += wt*c1
            if x_to_a is not None:
                x_to_a.append(arm_and_wts)
            if x_corrs is not None:
                x_corrs.append(c1)
            self.availability_array[x_idx] = 0
            
        return True
    
    def sample_from_arms(self, arm_wts, num_sample, counts0, counts1, sample_type="correctedwep"):
        """
        Sample `num_sample` examples weighted by arm_wts*(max_a Pr(a|x))
        :param arm_wts: list or array of weights indexed by self.dataset.arms to a float score
        """
        x_index_to_arm_indices, arm_index_to_indices = self.get_index_mappings(sample_type)
        noep_x_index_to_arm_indices, noep_arm_index_to_indices = self.get_index_mappings(sample_type="correctednoep")
        score_per_example = []
        for x_idx in range(len(noep_x_index_to_arm_indices)):
            arm_idx, wt = max(noep_x_index_to_arm_indices[x_idx], key=lambda _: _[1])
            score = self.availability_array[x_idx] * (np.log(wt) + np.log(arm_wts[arm_idx]))
            score_per_example.append(score)
        # pick the top ones
        top_x_idxs = np.argsort(score_per_example)[-num_sample:] 
        corrs = self.primary_model_evaluate_cached(top_x_idxs)["correct"]
        for x_idx, c1 in zip(top_x_idxs, corrs):
            arm_and_wts = x_index_to_arm_indices[x_idx]
            for ai, (arm_idx, wt) in enumerate(arm_and_wts):
                if wt > self.SAMPLE_TOL:
                    counts0[arm_idx] += wt*(1 - c1)
                    counts1[arm_idx] += wt*c1
            self.availability_array[x_idx] = 0            
        return True
    
    def arm_availability(self, sample_type):
        score_per_arm = np.zeros(self.dataset.num_arms)
        x_index_to_arm_indices, arm_index_to_indices = self.get_index_mappings(sample_type)
        for arm_idx in range(len(arm_index_to_indices)):
            x_idx_wts = arm_index_to_indices[arm_idx]
            x_idx_wts = [(x_idx, wt) for x_idx, wt in x_idx_wts if self.availability_array[x_idx]]
            if len(x_idx_wts) == 0:
                continue
            x_idx_wt = max(x_idx_wts, key=lambda _:_[1])
            score_per_arm[arm_idx] = x_idx_wt[1]
        return score_per_arm

    def sample_attribute2(self, preds: np.ndarray, node_potentials: np.ndarray, num_sample: int, use_edge_potentials=True):
        kernel_model = self.kernel_model
        if not use_edge_potentials:
            kernel_model = self.kernel_model_no_edge_potential
        with torch.no_grad():
            kernel_model.eval()
            potentials = kernel_model.potential(node_potentials, preds, use_edge_potentials)
        samples = []
        for k in range(len(potentials)):
            # potentials[k] -- len(np_x) x size of attr k
            _m = torch.min(potentials[k], axis=1)[0]
            _p = potentials[k] - torch.unsqueeze(_m, dim=1)
            print ("potential: ",_p, k)
            dist = torch.distributions.Multinomial(logits=_p)
            # batch_size x num_samples
            samples_per_attr = torch.stack([torch.nonzero(dist.sample(), as_tuple=False)[:, 1] for _ in range(num_sample)], dim=1)
            samples.append(samples_per_attr)
            
        # batch_size x num_samples x num_attrs
        samples = torch.stack(samples, dim=2).numpy()
        return samples
    
    def sample_attribute3(self, preds: np.ndarray, node_potentials: np.ndarray, num_sample: int, use_edge_potentials=True):
        kernel_model = self.kernel_model
        if not use_edge_potentials:
            kernel_model = self.kernel_model_no_edge_potential
        with torch.no_grad():
            kernel_model.eval()
            # n x top_k both of them
            arms, scores = kernel_model.potential2(node_potentials, preds, use_edge_potentials)
            scores -= torch.unsqueeze(torch.min(scores, dim=1)[0], 1)
            dist = torch.distributions.Multinomial(logits=scores)
            # num_sample x n x top_k
            sampled_indices = dist.sample((num_sample, ))
            # num_sample x n 
            sampled_indices = torch.max(sampled_indices, dim=-1)[1]
            sampled_indices = sampled_indices.t().type(torch.long)
        
        # n x num_sample
        sampled_arms = torch.gather(arms, index=sampled_indices, dim=1).type(torch.int64)
        attr_vecs = self.dataset.arms[sampled_arms]
        return attr_vecs
    
    def sample_attribute4(self, preds: np.ndarray, node_potentials: np.ndarray, num_sample: int, use_edge_potentials=True):
        """
        Just like sample_attribute3 but with the difference that we get rid of sampling completely and return the top arms along with their probabilities.
        Note the return type is different from any other sample_attribute*
        """
        kernel_model = self.kernel_model
        if not use_edge_potentials:
            kernel_model = self.kernel_model_no_edge_potential
        
        with torch.no_grad():
            kernel_model.eval()
            # n x top_k both of them
            arms, scores = kernel_model.potential2(node_potentials, preds, use_edge_potentials=use_edge_potentials, num_sample=num_sample)
            scores -= torch.unsqueeze(torch.min(scores, dim=1)[0], 1)
            probs = torch.softmax(scores, dim=1)
        
        return arms, probs
        
    def sample_attribute(self, np_x: np.ndarray, num_sample: int, debug=True, use_edge_potentials=True):
        """
        Return num_samples from fitted Pr(a|x)
        :x: input instances -- batch_size x DIM
        :num_sample: int
        Returns numpy array of shape: num_sample x batch_size x attr_dim  
        """
        preds = np.stack([model.preds(np_x) for model in self.models], axis=1)
        node_potentials = [model.probs(np_x) for model in self.models]
        
        if debug:
            print ("The predicted attribute values:", preds)
            print ("Logit scores on each attribute:")
            for _p in node_potentials:
                print (_p)
        
        return self.sample_attribute3(preds, node_potentials, num_sample, use_edge_potentials)
        
    def fit_acc_per_arm(self):
        """
        Computes emprical accuracy per arm
        """
        sys.stderr.write("Computing per arm accuracy\n")
        self.arm_index_to_acc = []
        
        np_x, np_y, _ = self.dataset.full_data()
        corrs = data_utils.cache(
            lambda: self.primary_model.evaluate(np_x, np_y)['correct'], 
            "%s/unlab_corrs.pkl" % self.cache_dir, 
            use_cache=self.cache_dir is not None
        )
        assert len(corrs) == len(np_x), "Unexpected!! Len of corrs: %d, input: %d" % (len(corrs), len(np_x))
        
        for ai in range(len(self.dataset.arms)):
            idxs = self.dataset.arm_to_idxs.get(ai, [])
            if len(idxs) < 5:
                arm_acc = np.nan
            else:
                arm_acc = corrs[idxs].sum()/float(len(idxs))
            self.arm_index_to_acc.append(arm_acc)

        vals = np.array(self.arm_index_to_acc)
        sys.stderr.write("Found %d unsupported arms of the total %d arms\n" % (len([_ for _ in vals if np.isnan(_)]), len(vals)))
        sys.stderr.write("Done\n")
        _vals = vals[np.where(vals>=0)]
        sys.stderr.write("Best, worst, mean accuracy: %0.4f, %0.4f %0.4f\n" % (np.max(_vals), np.min(_vals), np.mean(_vals)))
        sys.stderr.write("Acc quantiles: %s\n" % (str(np.quantile(_vals, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.]))))
        sys.stderr.write("Worst arms: %s\n" % str(np.sort(_vals)[:30]))
        
    def fit_acc_per_arm_validation(self):
        """
        Computes emprical accuracy per arm in the validation mode
        Which means the arm to acc mapping is computed only on the small labeled dataset
        """
        self.validation_arm_index_to_acc = []
        
        np_x, np_y, np_a = self.dataset.full_labeled_data()
        with open("%s/lab_probs.pkl" % self.cache_dir, "rb") as f:
            pred_y = np.argmax(pickle.load(f)[self.primary_task_index], axis=-1)
        corrs = np.equal(pred_y, np_y).astype(np.float32)
        assert len(corrs) == len(np_x), "Unexpected!! Len of corrs: %d, input: %d" % (len(corrs), len(np_x))
        
        arm_to_idxs = {}
        for xi in range(len(np_a)):
            idxs = arm_to_idxs.get(np_a[xi], [])
            idxs.append(xi)
            arm_to_idxs[np_a[xi]] = idxs
        
        for ai in range(len(self.dataset.arms)):
            idxs = arm_to_idxs.get(ai, [])
            if len(idxs) < 1:
                arm_acc = np.nan
            else:
                arm_acc = corrs[idxs].sum()/float(len(idxs))
            self.validation_arm_index_to_acc.append(arm_acc)
        
    def area(self, lst, stds, start=0.):
        gap = 0.001
        tols = np.arange(start, 1., gap)
        ret = 0
        lst = np.array(lst)
        for tol in tols:
            num = np.where(lst<tol)[0].size
            frac = num/len(lst)
            ret += (1 - frac)
        return ret*100*gap
    
    # Micro-averaged MSE
    def evaluate_mean_micromse(self, estimated_arm_index_to_acc):
        diffs = [abs(self.arm_index_to_acc[ai] - estimated_arm_index_to_acc[ai]) for ai in range(len(self.arm_index_to_acc))]
            
        num_arms = 0
        active_idxs = [ai for ai in range(len(self.arm_index_to_acc)) if not np.isnan(self.arm_index_to_acc[ai])]
        freqs = np.array([len(self.dataset.arm_to_idxs[arm_idx]) for arm_idx in active_idxs]).astype(np.float32)
        freqs /= freqs.sum()
        ret = np.sum(freqs*(np.array(diffs)[active_idxs]**2))
        return ret

    # evaluate on the worst 80 arms -- worst: arms with accuracy close to 0
    def evaluate_mean_worstacc(self, estimated_arm_index_to_acc):
        diffs = [abs(self.arm_index_to_acc[ai] - estimated_arm_index_to_acc[ai]) for ai in range(len(self.arm_index_to_acc))]
        
        worst_idxs = [(ai, _acc) for ai, _acc in enumerate(self.arm_index_to_acc) if not np.isnan(_acc)]
        num = 50
        sidxs = sorted(worst_idxs, key=lambda _: _[1])
        worst_idxs = [ai for ai, acc in sidxs[:num]]
        ret = np.sum((np.array(diffs)[worst_idxs]**2))/len(worst_idxs)
        
        all_errs = []
        active_idxs = [(ai, _acc) for ai, _acc in enumerate(self.arm_index_to_acc) if not np.isnan(_acc)]
        for ai, _ in active_idxs:
            _err = {
                'true': self.arm_index_to_acc[ai],
                'diff': diffs[ai],
                'freq': len(self.dataset.arm_to_idxs[ai])
            }
            all_errs.append(_err)
        return ret, all_errs
    
    # evaluate on 80 least frequent arms
    def evaluate_mean_worstfreq(self, estimated_arm_index_to_acc):
        diffs = [abs(self.arm_index_to_acc[ai] - estimated_arm_index_to_acc[ai]) for ai in range(len(self.arm_index_to_acc))]
        
        worst_idxs = [ai for ai, _acc in enumerate(self.arm_index_to_acc) if not np.isnan(_acc)]
        worst_idxs = [(arm_idx, len(self.dataset.arm_to_idxs[arm_idx])) for arm_idx in worst_idxs]
        sw = sorted(worst_idxs, key=lambda _: _[1])
        num = 50
        worst_idxs = [ai for ai, ln in sw[:num]]
        ret = np.sum((np.array(diffs)[worst_idxs]**2))/len(worst_idxs)
        return ret
    
    # micro-averaged MSE
    def evaluate_mean_estimates(self, estimated_arm_index_to_acc, acc_sigma=0.0, debug=False):
        # there could be arms without any examples, do not evaluate on sych arms
        diffs = [abs(self.arm_index_to_acc[ai] - estimated_arm_index_to_acc[ai]) for ai in range(len(self.arm_index_to_acc))]
            
        num_arms = 0
        worst_idxs = []
        for ai in range(len(self.arm_index_to_acc)):
            _acc = self.arm_index_to_acc[ai]
            if np.isnan(_acc):
                continue
            num_arms += 1
            if diffs[ai] >= acc_sigma:
                worst_idxs.append(ai)
        ret = [np.sum(np.array(diffs)[worst_idxs]**2)/num_arms]
        if debug:
            ret.append(diffs)
        return ret
        
    def _approximate_softmax_loss(self, node_potentials, preds, idxs, use_edge_potentials, add=True):
        if use_edge_potentials:
            kernel_model = self.kernel_model
        else:
            kernel_model = self.kernel_model_no_edge_potential
        
        batch_size = len(idxs)
        num_attrs = np.shape(preds)[-1]
        loss = 0
        potentials = kernel_model.potential(
            [node_potential[idxs] for node_potential in node_potentials], 
            preds[idxs], 
            use_edge_potentials=use_edge_potentials, 
            add=add
        )
        for k in range(num_attrs):
            # batch_size x vocab size of this attr
            potential_this_attr = potentials[k]
            attr_k_size = potential_this_attr.shape[-1]
            potential_this_attr -= torch.unsqueeze(torch.min(potential_this_attr, dim=1)[0], 1)
            pred_attr = preds[idxs, k]

            this_loss = torch.logsumexp(potential_this_attr, dim=1) - potential_this_attr[np.arange(batch_size), pred_attr]
            assert list(this_loss.shape) == [len(potential_this_attr)]
            loss += torch.sum(this_loss)
        loss /= num_attrs
        
        return loss
    
    def _softmax_loss(self, node_potentials, preds, idxs, use_edge_potentials):
        if use_edge_potentials:
            kernel_model = self.kernel_model
            edge_scores = kernel_model(torch.tensor(self.dataset.arms).type(torch.float)).squeeze()
        else:
            kernel_model = self.kernel_model_no_edge_potential
        
        pred_ais = [self.dataset.hash_to_arm_index(self.dataset.hash_arm(preds[ii])) for ii in idxs]
        for pred_ai in pred_ais:
            assert pred_ai is not None
                
        node_scores = torch.zeros([len(idxs), self.dataset.num_arms])
        num_attrs = preds.shape[-1]
        for ai in range(num_attrs):
            # len(idxs) x num_arms
            np_this_attr = torch.tensor(node_potentials[ai][idxs])
            node_scores += np_this_attr[:, self.dataset.arms[:, ai]] * torch.exp(kernel_model.temp_param[ai])
            
#         node_scores = node_scores - torch.logsumexp(node_scores, dim=1).unsqueeze(-1)
        _scores = torch.zeros_like(node_scores)
        _scores = node_scores
        if use_edge_potentials:
            edge_scores = torch.unsqueeze(edge_scores, dim=0)
#             edge_scores = edge_scores - torch.unsqueeze(torch.logsumexp(edge_scores, dim=-1), dim=-1)
            if kernel_model.training:
                mask = np.random.binomial(1, 0.5, [len(idxs), 1])
                mask = torch.tensor(mask)
                _scores = node_scores + mask*edge_scores
            else:
                _scores = node_scores + edge_scores
            
        # 10 neg. samples
#         neg_sample_idxs = np.stack([np.random.choice(self.dataset.num_arms, size=[10], replace=False) for _ in range(len(idxs))], axis=0)
#         all_idxs = np.concatenate([neg_sample_idxs, np.reshape(pred_ais, [-1, 1])], axis=1)
#         # ref on adv. indexing: https://stackoverflow.com/questions/37878946/indexing-one-array-by-another-in-numpy
#         loss = torch.logsumexp(_scores[np.arange(len(idxs))[:, None], all_idxs], dim=1) - _scores[np.arange(len(idxs)), pred_ais]
#         loss = torch.mean(loss)
        loss = torch.mean(torch.logsumexp(_scores, dim=1) - _scores[np.arange(len(idxs)), pred_ais])

        return loss
        
    def fit(self, use_edge_potentials=True):
        """
        Fits a kernel modeling edge potentials by optimizing data likelihood
        """
        # not to use any of the labels from the unlabeled set
        np_x, _, _ = self.dataset.full_data()
        l_np_x, l_np_y, l_np_arm_ids = self.dataset.full_labeled_data()
        l_np_a = self.dataset.arms[l_np_arm_ids]
    
        # obtain predictions and probs for full data
        # [ len(np_x)x(num values in this attr) ]_{xnum_attrs}
        if self.joint_model is None:
            node_potentials = data_utils.cache(
                lambda: [model.probs(np_x, debug=True) for model in self.models], 
                "%s/unlab_probs.pkl" % self.cache_dir,
                use_cache=self.cache_dir is not None
            )
        else:
            sys.stderr.write ("Using joint model.\n")
            node_potentials = data_utils.cache(
                lambda: self.joint_model.logit_per_attr(np_x, debug=True), 
                "%s/unlab_probs.pkl" % self.cache_dir,
                use_cache=self.cache_dir is not None
            )
        preds = [np.argmax(_np, axis=-1) for _np in node_potentials]
        
        if self.joint_model is None:
            l_node_potentials = data_utils.cache(
                lambda: [model.probs(l_np_x, debug=True) for model in self.models], 
                "%s/lab_probs_seed=%d.pkl" % (self.cache_dir, self.dataset.seed),
                use_cache=self.cache_dir is not None
            )
        else:
            sys.stderr.write ("Using joint model.\n")
            l_node_potentials = data_utils.cache(
                lambda: self.joint_model.logit_per_attr(l_np_x, debug=True), 
                "%s/lab_probs_seed=%d.pkl" % (self.cache_dir, self.dataset.seed),
                use_cache=self.cache_dir is not None
            )
        for ai in range(len(l_node_potentials)):
            _p = np.argmax(l_node_potentials[ai], axis=-1)
            acc = (np.equal(_p, l_np_a[:, ai])).astype(np.float32).mean()
            print ("Labeled ai: %d acc: %0.4f" % (ai, acc))
        
        # len(np_x) x num_attrs
        preds = np.stack(preds, axis=1)
        num_attrs = np.shape(preds)[-1]
        print ("Len:", len(preds))
        print (preds[:10, 0], node_potentials[0][:10])
        print ("Shape: ", np.shape(l_np_a))
        
        if use_edge_potentials:
            kernel_model = self.kernel_model
        else:
            kernel_model = self.kernel_model_no_edge_potential
        
        # parameters
        kernel_model.train()
#         optimizer = torch.optim.SGD(kernel_model.parameters(), lr=1e-2, momentum=0., weight_decay=1e-2)
        optimizer = torch.optim.Adam(kernel_model.parameters(), lr=1e-3)
        for param in kernel_model.parameters():
            torch.nn.init.normal_(param, 0, 1)
        
        np.random.seed(1)
        unlabeled_indices = np.arange(len(np_x))
        np.random.shuffle(unlabeled_indices)
        split = int(len(unlabeled_indices)*0.8)
        unlabeled_train_indices, unlabeled_test_indices = unlabeled_indices[:split], unlabeled_indices[split:]
        
#         pred_unlab_train_arm_indices = np.array([self.dataset.hash_to_arm_index(self.dataset.hash_arm(_p)) for _p in preds[unlabeled_train_indices]])
#         u, c = np.unique(pred_unlab_train_arm_indices, return_counts=True)
#         count_dict = dict(zip(u, c))
#         print ("Most common arms: ", list(count_dict.items())[:20])
#         count = np.array([count_dict[_p] for _p in pred_unlab_train_arm_indices])
#         inv_count = 1./count
#         inv_count /= inv_count.sum()
                
        np.random.seed(1)
        labeled_indices = np.arange(len(l_np_x))
        np.random.shuffle(labeled_indices)
        split = int(len(labeled_indices)*0.8)
        labeled_train_indices, labeled_test_indices = labeled_indices[:split], labeled_indices[split:]
        
        model_name = self.model_name
        if not use_edge_potentials:
            model_name = self.model_name_no_edge_potential
        
        batch_size, nepochs = 128, 100
        nsteps = (len(np_x)//batch_size)*nepochs
        best_val, prev_val_loss = 1e10, 1e10
        strikes, TOL = 0, 20
        _, _, np_unlabeled_arm_indices = self.dataset.full_data()
        np_unlabeled_a = self.dataset.arms[np_unlabeled_arm_indices]
#         loss_fn = self._approximate_softmax_loss
        loss_fn = self._softmax_loss
        for nstep in tqdm.tqdm(range(nsteps), desc="Kernel fitting"):
            # set loss to maximize difference between \Sigma_{j=1}^d log Pr(a_i|x) - \Sigma_v log Pr(a_i-j,v|x) expression (7)
            loss = 0
            # Add to loss unlabeled and labeled
            for _ in [0, 1]: 
                if _ == 0:
                    idxs = np.random.choice(unlabeled_train_indices, batch_size)
                    _node_potentials = node_potentials
                    _preds = np_unlabeled_a
                    # _preds = preds
                else:
                    idxs = np.random.choice(labeled_train_indices, batch_size)
                    _node_potentials = l_node_potentials
                    _preds = l_np_a

                loss += loss_fn(_node_potentials, _preds, idxs, use_edge_potentials=use_edge_potentials)

            optimizer.zero_grad()
            loss.backward()
            if str(self.dev) == 'cpu':
                optimizer.step()
            else:
                xm.optimizer_step(optimizer, barrier=True)
                        
            if nstep%100 == 0:
                sys.stderr.write ("Temperatures: %s\n" % (str(np.exp(kernel_model.temp_param.detach().cpu().numpy()))))
                kernel_model.eval()
                train_loss = loss_fn(l_node_potentials, l_np_a, labeled_train_indices, use_edge_potentials=use_edge_potentials)
                val_loss = loss_fn(l_node_potentials, l_np_a, labeled_test_indices, use_edge_potentials=use_edge_potentials)
                val_loss2 = loss_fn(node_potentials, np_unlabeled_a, unlabeled_test_indices, use_edge_potentials=use_edge_potentials)
                sys.stderr.write ("%d/%d steps finished strikes: %d/%d Loss: %0.3f Train loss: %0.3f Val loss: %0.3f %0.3f\n" % (nstep, nsteps, strikes, TOL, loss.detach().cpu().numpy(), train_loss.detach().cpu().numpy(), val_loss.detach().cpu().numpy(), val_loss2.item()))
                kernel_model.train()
                if val_loss < best_val:
                    best_val = val_loss
                    sys.stderr.write("Saving model to %s\n" % model_name)
                    torch.save(kernel_model.cpu().state_dict(), model_name)
                if val_loss >= prev_val_loss:
                    strikes += 1
                prev_val_loss = val_loss
            if strikes > TOL:
                break
        
        kernel_model.load_state_dict(torch.load(model_name))
        kernel_model.eval()
        train_loss = loss_fn(l_node_potentials, l_np_a, labeled_train_indices, use_edge_potentials=use_edge_potentials)
        # Measure test loss on unlabeled data since the labeled test could be too small
        test_loss2 = loss_fn(node_potentials, np_unlabeled_a, unlabeled_test_indices, use_edge_potentials=use_edge_potentials)
        test_loss = loss_fn(l_node_potentials, l_np_a, labeled_test_indices, use_edge_potentials=use_edge_potentials)
        
        print ("Train, Test loss on labeled data: %0.3f %0.3f %0.3f" % (train_loss, test_loss, test_loss2))
        
def simple_estimator(args, kwargs):
    dataset, data_fitter, cache_dir = args[0], args[1], args[2]
    for num_sample in [0, 500, 1000, 1500, 2000, 3000, 4000, 5000]: 
        _errs = []
        for seed in range(3):
            np.random.seed(seed)
            kwargs['seed'] = seed
            # set seed for the dataset
            args[0].seed = seed
            exp = BetaGPExplorer(*args, **kwargs)
            status = exp.fitter.sample(num_sample, exp.counts0, exp.counts1, exp.explored, sample_type=exp.sample_type)
            
            print ("Counts: ", exp.counts0.sum(), exp.counts1.sum())
            mu_hat = exp.simple_fit()
            err = exp.fitter.evaluate_mean_estimates(mu_hat)[0]
            print (err)
            print ("NS: %d Seed: %d err: %f" % (num_sample, seed, err))
            _errs.append(err)
        m, s = np.mean(_errs), np.std(_errs)
        print (m, s)
        errs[num_sample] = (m, s)

    