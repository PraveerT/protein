import numpy as np
import os, sys, h5py
from torch.utils.data import Dataset
import torch
from .build import DATASETS
from utils.logger import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


@DATASETS.register_module()
class ProteinEC(Dataset):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.subset = config.subset
        self.root = config.ROOT
        
        if self.subset == 'train':
            h5_path = os.path.join(self.root, 'protein_train.h5')
        elif self.subset == 'val':
            h5_path = os.path.join(self.root, 'protein_val.h5')
        elif self.subset == 'test':
            h5_path = os.path.join(self.root, 'protein_test.h5')
        else:
            raise NotImplementedError()

        self.h5_file = h5py.File(h5_path, 'r')
        self.coords = self.h5_file['coords']
        self.features = self.h5_file['labels']
        self.masks = self.h5_file['mask']
        self.targets = self.h5_file['targets']

        print(f'Successfully loaded ProteinEC {self.subset} set. Shape of coords: {self.coords.shape}')

    def __getitem__(self, idx):
        # Get data from H5 files
        current_coords = self.coords[idx]   # Shape: (1024, 3)
        current_features = self.features[idx] # Shape: (1024,)
        current_mask = self.masks[idx]     # Shape: (1024,)
        target_label = self.targets[idx]   # Scalar

        # Combine coordinates and amino acid features into a single tensor
        combined_points = np.concatenate(
            [current_coords, current_features[..., np.newaxis]], 
            axis=-1
        ).astype(np.float32)

        # Augment training set
        if self.subset == 'train':
            pt_idxs = np.arange(0, combined_points.shape[0])
            np.random.shuffle(pt_idxs)
            combined_points = combined_points[pt_idxs]
            current_mask = current_mask[pt_idxs]

        # Convert to Tensor
        points_tensor = torch.from_numpy(combined_points).float()
        mask_tensor = torch.from_numpy(current_mask).bool()
        target_tensor = torch.tensor(target_label).long()

        dummy_id = "protein_" + str(idx)
        
        # Return new tuples
        return dummy_id, dummy_id, (points_tensor, mask_tensor, target_tensor)

    def __len__(self):
        return len(self.targets)
    
    def close(self):
        self.h5_file.close()
