import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parent.parent))
from constants import DATA_PATH


class TrainDataset(Dataset):
    def __init__(self):
        self.data = np.load(DATA_PATH + '\\stock_inputs.npy')
        self.label = np.load(DATA_PATH + '\\stock_labels.npy')
        self.train_len = self.data.shape[0]

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],int(self.data[index,0,-1]), self.label[index])

class TestDataset(Dataset):
    def __init__(self):
        self.data = np.load(DATA_PATH + '\\stock_test_inputs.npy')
        self.label = np.load(DATA_PATH + '\\stock_test_labels.npy')
        self.test_len = self.data.shape[0]
        
    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],int(self.data[index,0,-1]),self.label[index])
