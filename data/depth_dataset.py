### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import h5py
import numpy as np
import torch


class DepthDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.data_dir = os.path.join(opt.dataroot, 'train')
        # self.A_paths = sorted(make_dataset(self.dir_A))
        self.data_paths = []
        self.dataset_size = 0
        assert os.path.isdir(self.data_dir), '%s is not a valid directory' % self.data_dir

        for root, _, fnames in sorted(os.walk(self.data_dir)):
            for fname in fnames:
                path = os.path.join(root, fname)
                self.data_paths.append(path)
                self.dataset_size += 1

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        try:
            h5f = h5py.File(data_path, "r")
        except OSError:
            return dict()
        rgb = np.array(h5f['rgb'])
        #rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.array(h5f['depth'])
        depth = np.dstack((depth, depth, depth))
        depth = np.transpose(depth, (2, 0, 1))
        rgb = torch.tensor(rgb, dtype=torch.float)
        depth = torch.tensor(depth, dtype=torch.float)
        input_dict = {'label': rgb, 'inst': 0, 'image': depth,
                      'feat': 0, 'path': data_path}
        #print("=" * 10)
        #print(rgb.shape)
        #print(depth.shape)
        #print(rgb)
        #print(depth)
        return input_dict

    def __len__(self):
        return len(self.data_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'DepthDataset'
