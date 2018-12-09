import numpy as np
import h5py
import os
from data.dense_to_sparse import SimulatedReflector


data_dir = '../datasets/NYUdepth'
output_folder = 'NYUdepth_d'

def modify(data_path):
    #data_path = data_paths[index][0]
    h5f = h5py.File(data_path, "r+")
    rgb = np.array(h5f['rgb'])
    depth = np.array(h5f['depth'])
    depth = np.dstack((depth, depth, depth))
    rgbd = create_sparse_depth(rgb, depth)
    rgbd = np.transpose(rgbd, (2, 0, 1))
    h5f['rgbd'] = rgbd
    print('File %s modified.'%data_path)
    h5f.close()

def create_sparse_depth(self, rgb, depth):
    sparsifier = SimulatedReflector(num_samples=self.opt.num_samples, max_depth=max_depth)
    rgb = np.transpose(rgb, (1,2,0))
    depth = depth[:, :, 0]
    mask_keep = sparsifier.dense_to_sparse(rgb, depth)
    sparse_depth = np.zeros(depth.shape)
    sparse_depth[mask_keep] = depth[mask_keep]
    rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
    return rgbd

data_paths = []
output_folder = []
for root, dirnames, fnames in sorted(os.walk(data_dir)):
    for fname in fnames:
        path = os.path.join(root, fname)
        #folders = root.split('/')
        #out_path = folders[:2] + ['NYUdepth_d'] + folders[3:] + [fname]
        #data_paths.append((path, '/'.join(out_path)))
        modify(path)