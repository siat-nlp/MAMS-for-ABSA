import torch
from torch.utils.data import Dataset
import numpy as np

class ABSADataset(Dataset):

    def __init__(self, path, input_list):
        super(ABSADataset, self).__init__()
        data = np.load(path)
        self.data = {}
        for key, value in data.items():
            self.data[key] = torch.tensor(value).long()
        self.len = self.data['label'].size(0)
        self.input_list = input_list

    def __getitem__(self, index):
        return_value = []
        for input in self.input_list:
            return_value.append(self.data[input][index])
        return_value.append(self.data['label'][index])
        return return_value

    def __len__(self):
        return self.len