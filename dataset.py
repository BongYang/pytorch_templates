import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, opt):
        pass
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)
    
    def __len__(self):
        return 0
    
class TestDataset(Dataset):
    def __init__(self, opt):
        pass
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)
    
    def __len__(self):
        return 0