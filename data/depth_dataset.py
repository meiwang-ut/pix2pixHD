### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import h5py
import numpy as np
import torch
import cv2


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
        depth = np.array(h5f['depth'])
        depth = np.dstack((depth, depth, depth))
        depth = np.transpose(depth, (2, 0, 1))  # chanel first
        if self.opt.sparse:
            sparse_depth = self.create_sparse_depth(depth)
            rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        rgb = torch.tensor(rgb, dtype=torch.float)
        depth = torch.tensor(depth, dtype=torch.float)
        input_dict = {'label': rgb, 'inst': 0, 'image': depth,
                      'feat': 0, 'path': data_path}
        if self.opt.sparse:
            input_dict['label'] = rgbd
        return input_dict

    def __len__(self):
        return len(self.data_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'DepthDataset'

    def create_sparse_depth(self, rgb, depth):
        mask_keep = self.dense_to_sparse(rgb, depth)
        sparse_depth = np.zeros(depth.shape)
        sparse_depth[mask_keep] = depth[mask_keep]

        return sparse_depth

    def dense_to_sparse(self, rgb, depth, max_depth=0.0, dilate_kernel=3, dilate_iterations=1):
        gray = self.rgb2grayscale(rgb)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)

        depth_mask = np.bitwise_and(depth != 0.0, depth <= max_depth)

        edge_fraction = float(self.num_samples) / np.size(depth)

        mag = cv2.magnitude(gx, gy)
        min_mag = np.percentile(mag[depth_mask], 100 * (1.0 - edge_fraction))
        mag_mask = mag >= min_mag

        if dilate_iterations >= 0:
            kernel = np.ones((dilate_kernel, dilate_kernel), dtype=np.uint8)
            cv2.dilate(mag_mask.astype(np.uint8), kernel, iterations=dilate_iterations)

        mask = np.bitwise_and(mag_mask, depth_mask)
        return mask

    def rgb2grayscale(self, rgb):
        return rgb[:, :, 0] * 0.2989 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114

