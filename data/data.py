from glob import glob
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import utils.utils as utils



def load_data_file(DATA_DIR, partition):		

    data_dir = os.path.join(DATA_DIR, partition)
    all_data = glob(os.path.join(data_dir, '*.txt'))

    data_batchlist, label_batchlist = [], []
    for f in all_data:   
        data = np.loadtxt(f, delimiter=',')
        label = np.loadtxt(f[:-4] + '.labels')
        data_batchlist.append(data)
        label_batchlist.append(label)

    unique, counts = np.unique(np.hstack(label_batchlist), return_counts=True)
    label_weights = counts.astype(np.float32)
    label_weights = label_weights / np.sum(label_weights)
    label_weights = 1 / np.log(1.2 + label_weights)
    return data_batchlist, label_batchlist, label_weights
    
class Dataset(Dataset):
    def __init__(self, num_pts, DATA_DIR, partition, tile_size=30):	
        super().__init__() 
        self.partition = partition
        self.num_pts = num_pts
        self.tile_size = tile_size
        self.data, self.seg, self.class_weights = load_data_file(DATA_DIR, partition)

    def __getitem__(self, idx):
        pt_idxs = np.random.choice(len(self.seg[idx]), self.num_pts)
        pts = self.data[idx][pt_idxs]

        pts[:, 0:3] = self._center_box(pts[:, 0:3])
        
        pc_tensor_convertor = utils.PCToTensor()
        rotator = utils.PCRotate()
        current_pts = rotator(pc_tensor_convertor(pts))
        
        current_labels = torch.from_numpy(self.seg[idx][pt_idxs].copy()).long()
        return current_pts, current_labels, self.class_weights

    def __len__(self):
        return len(self.data)
        
    def _center_box(self, pts):
        box_min = np.min(pts, axis=0)
        shift = np.array([box_min[0] + self.tile_size / 2, box_min[1] + self.tile_size / 2, box_min[2]])
        pts_centered = pts - shift
        return pts_centered
