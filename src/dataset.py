from abc import ABC, abstractmethod
from .utils import data_utils, config, train_utils
import numpy as np
import torch
import tqdm
import time

if config.xla:
    import torch_xla.core.xla_model as xm

class Dataset(ABC):
    """
    Routines for dataset loading and sampling
    """
            
    def full_data(self):
        """
        Returns all the available data
        :return: np.arrays of (x, y, arm_indices)
        same interpretation of notation as self.sample
        """
        pass
    
    def full_data_arm(self, arm_index):
        """
        Returns all the available data for that arm.
        Returns x, y numpy array
        """
        pass
    
    @property
    def arms(self):
        """
        :return: np array of arms of size len(D) x num attributes
        """
        return self.arms
    
    @property
    def num_arms(self):
        return len(self.arms)
    
    def num_attrs(self):
        """
        :return: number of attributes 
        """
        return np.shape(self.arms)[-1]

    @abstractmethod
    def __len__(self):
        """
        :return: length of available data
        """
        pass
    
    @abstractmethod
    def hash_arm(self, arm_index) -> str:
        """
        :return: signature string of arm with arm_index
        """
        pass
    
    @abstractmethod
    def hash_to_arm_index(self, hashed_arm: str):
        """
        Returns an index in to self.arms
        """
        pass
    

class Model(ABC):
    def __init__(self, torch_model, device, x_transform=None):
        """
        :args: Torch model and device, the forward of model should give logits(classification)/scalar(regression)
        :x_transform: Any needed transform before we obtain predictions.
        """
        self.model = torch_model.to(device)
        self.model.eval()
        self.dev = device
        self.transform = x_transform
        
    def evaluate(self, np_x, np_y, status=True):
        """
        Evaluates self on the provided data
        """
        arm_dataset = data_utils.DatasetNumpy(np_x, np_y, transform=self.transform)
        arm_loader = torch.utils.data.DataLoader(arm_dataset, batch_size=128)
        
        return train_utils.evaluate(self.model, arm_loader, self.dev, status=status)
    
    def preds(self, np_x, debug=False):
        """
        :return: Predictions of the model on input
        """        
        dataset = data_utils.DatasetNumpy(np_x, np.zeros([len(np_x)]), transform=self.transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=512)
        preds_list = []
        st = time.time()
        img_time, fetch_time, move_time = 0, 0, 0
        step = 0
        if debug:
            loader = tqdm.tqdm(loader, desc="Predictions...")
        with torch.no_grad():
            for x, _ in loader:
                img_time += time.time() - st
                st = time.time()
                x = x.to(self.dev)
                move_time += time.time() - st
                st = time.time()
                preds = self.model(x)
                preds = torch.argmax(preds, dim=1)
                preds_list.append(preds.detach().cpu().numpy())
                fetch_time += time.time() - st
                
                step += 1
#                 if step % 10 == 0:
#                     print ("Img time: %0.4f fetch time: %0.4f move time: %0.4f" % (img_time, fetch_time, move_time))
                    
                st = time.time()
            np_preds = np.concatenate(preds_list, axis=0)
            return np_preds
        
    def probs(self, np_x, debug=False):
        """
        Returns \propto Pr(y|x) for each y
        shape of return array [len(np_x) x num output]
        """        
        dataset = data_utils.DatasetNumpy(np_x, np.zeros([len(np_x)]), transform=self.transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128)
        probs_list = []
        if debug: 
            loader = tqdm.tqdm(loader, desc='Probs..')
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.dev)
                preds = self.model(x)
                np_preds = preds.detach().cpu().numpy()
                probs_list.append(np_preds)
            np_probs = np.concatenate(probs_list, axis=0)
            return np_probs
        
class ModelFromCache(Model):
    """
    Initialized a model like beavour giving preds and probs, by reading from cache.
    The np_x param to evaluate and preds should be indices in to logits, truth.
    """
    def __init__(self, logits):
        """
        :arg probs: data map from index to logit scores for each index
        """
        self.logits = logits
        
    def evaluate(self, np_x, np_y, status=True):
        """
        Evaluates self on the provided data
        """
        preds = self.preds(np_x)
        corrs = (preds == np_y).astype(np.float32)
        return {"correct": corrs, "acc": corrs.mean()}
    
    def preds(self, np_x, debug=False):
        """
        :return: Predictions of the model on input
        """        
        logits = self.logits[np_x]
        preds = np.argmax(logits, axis=-1)
        return preds
        
    def logits(self, np_x, debug=False):
        """
        Returns \propto Pr(y|x) for each y
        shape of return array [len(np_x) x num output]
        """   
        return self.logits[np_x]
    
    def probs(self, np_x, debug=False):
        """
        Returns \propto Pr(y|x) for each y
        shape of return array [len(np_x) x num output]
        """   
        return self.logits[np_x]
