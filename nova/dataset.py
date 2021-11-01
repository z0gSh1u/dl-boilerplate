'''
    Dataset interface.
'''

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import os
import numpy as np


class YOUR_DATASET(Dataset):
    def __init__(self, dir_) -> None:
        """Your dataset interface.
        """
        super(YOUR_DATASET, self).__init__()
        self.dir = dir_
        self.images = []  # fill it
        # Add data transforms here.
        self.transforms = transforms.Compose([])

    def __getitem__(self, index):
        # Return the data at index.
        return self.transforms(self.images[index])

    def __len__(self):
        return len(self.images)
