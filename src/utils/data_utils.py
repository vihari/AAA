import numpy as np
from PIL import Image
import torch
import pickle
import os, sys

class DatasetNumpy(torch.utils.data.Dataset):
    def __init__(self, np_x, np_y, transform=None):
        self.np_x = np_x
        self.np_y = np_y
        self.transform = transform
        
    def __getitem__(self, index):
        x = Image.fromarray(self.np_x[index]).convert("RGB")
        y = self.np_y[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.np_x)


def cache(fn, fname, use_cache: bool=True, verbose=False):
    if os.path.exists(fname) and use_cache:
        if verbose:
            sys.stdout.write("Using cached file: %s\n" % fname)
        with open(fname, "rb") as f:
            obj = pickle.load(f)
            return obj
    else:
        obj = fn()
        with open(fname, "wb") as f:
            pickle.dump(obj, f)
        return obj