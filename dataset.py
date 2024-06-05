import logging
import os
import re

import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import h5py
import math
import transforms3d
import random
from tensorpack import dataflow

from scipy.spatial.distance import cdist


def resample_pcd(pcd, n):
    """
    Randomly samples n points from a point cloud, repeating some as needed to reach n.
    Args:
        pcd: point cloud
        n: number of sampled points

    Returns: A point cloud with n points, and the indices used from the original pcd.
    """
    pcd_n = pcd.shape[0]
    idx = np.random.permutation(pcd_n)
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd_n, size=n - pcd_n)])
    return pcd[idx[:n]], idx[:n]


logger = logging.getLogger(__name__)


class PCN_pcd(data.Dataset):
    def __init__(self, path, prefix="train"):
        if prefix=="train":
            self.file_path = os.path.join(path,'train') 
        elif prefix=="val":
            self.file_path = os.path.join(path,'val')  
        elif prefix=="test":
            self.file_path = os.path.join(path,'test') 
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix
        self.label_map ={'02691156': '0', '02933112': '1', '02958343': '2',
                         '03001627': '3', '03636649': '4', '04256520': '5',
                         '04379243': '6', '04530566': '7', 'all': '8'}
        
        self.label_map_inverse ={'0': '02691156', '1': '02933112', '2': '02958343',
                         '3': '03001627', '4': '03636649', '5': '04256520',
                         '6': '04379243', '7': '04530566', '8': 'all'}

        self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))

        random.shuffle(self.input_data)

        self.len = len(self.input_data)

        self.scale = 0
        self.mirror = 1
        self.rot = 0
        self.sample = 1

    def __len__(self):
        return self.len

    def read_pcd(self, path):
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points)
        return points

    def get_data(self, path):
        cls = os.listdir(path)
        data = []
        labels = []
        for c in cls:
            objs = os.listdir(os.path.join(path, c))
            for obj in objs:
                f_names = os.listdir(os.path.join(path, c, obj))
                obj_list = []
                for f_name in f_names:
                    data_path = os.path.join(path, c, obj, f_name)
                    obj_list.append(data_path)
                    # points = self.read_pcd(os.path.join(path, c, obj, f_name))
                data.append(obj_list)
                labels.append(self.label_map[c])


        return data, labels

    def randomsample(self, ptcloud ,n_points):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:n_points]]

        if ptcloud.shape[0] < n_points:
            zeros = np.zeros((n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])
        return ptcloud

    def upsample(self, ptcloud, n_points):
        curr = ptcloud.shape[0]
        need = n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            # ptcloud = np.concatenate([ptcloud,np.zeros_like(ptcloud)],dim=0)
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud


    def get_transform(self, points):
        result = []
        rnd_value = np.random.uniform(0, 1)

        if self.mirror and self.prefix == 'train':
            trfm_mat = transforms3d.zooms.zfdir2mat(1)
            trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
            trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
            if rnd_value <= 0.25:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
            elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            elif rnd_value > 0.5 and rnd_value <= 0.75:
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        for ptcloud in points:
            ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)

            if self.scale:
                ptcloud = ptcloud * self.scale
            result.append(ptcloud)

        return result[0],result[1]

    def __getitem__(self, index):

        partial_path = self.input_data[index]
        n_sample = len(partial_path)
        idx = random.randint(0, n_sample-1)
        partial_path = partial_path[idx]

        partial = self.read_pcd(partial_path)

        # if self.prefix == 'train' and self.sample:
        partial = self.upsample(partial, 2048)

        gt_path = partial_path.replace('/'+partial_path.split('/')[-1],'.pcd')
        gt_path = gt_path.replace('partial','complete')


        if self.prefix == 'train':
            complete = self.read_pcd(gt_path)
            partial, complete = self.get_transform([partial, complete])
        else:
            complete = self.read_pcd(gt_path)

        complete = torch.from_numpy(complete)
        partial = torch.from_numpy(partial)
        label = partial_path.split('/')[-3]
        label = self.label_map[label]
        obj = partial_path.split('/')[-2]
        
        if self.prefix == 'test':
            return label, partial, complete, obj
        else:
            return label, partial, complete


class C3D_h5(data.Dataset):
    def __init__(self, path, prefix="train"):
        if prefix=="train":
            self.file_path = os.path.join(path,'train') 
        elif prefix=="val":
            self.file_path = os.path.join(path,'val')  
        elif prefix=="test":
            self.file_path = os.path.join(path,'test') 
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix
        self.label_map ={'02691156': '0', '02933112': '1', '02958343': '2',
                         '03001627': '3', '03636649': '4', '04256520': '5',
                         '04379243': '6', '04530566': '7', 'all': '8'}

        if prefix != "test":
            self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))
            self.gt_data, _ = self.get_data(os.path.join(self.file_path, 'gt'))
            print(len(self.gt_data), len(self.labels))
        else:
            self.input_data, self.labels = self.get_data(os.path.join(self.file_path, 'partial'))

        print(len(self.input_data))

        self.len = len(self.input_data)

        self.scale = 1
        self.mirror = 1
        self.rot = 0
        self.sample = 1

    def __len__(self):
        return self.len

    def get_data(self, path):
        cls = os.listdir(path)
        data = []
        labels = []
        for c in cls:
            objs = os.listdir(os.path.join(path, c))
            for obj in objs:
                data.append(os.path.join(path,c,obj))
                if self.prefix == "test":
                    labels.append(obj)
                else:
                    labels.append(self.label_map[c])

        return data, labels


    def get_transform(self, points):
        result = []
        rnd_value = np.random.uniform(0, 1)
        angle = random.uniform(0,2*math.pi)
        scale = np.random.uniform(1/1.6, 1)

        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        if self.mirror and self.prefix == 'train':

            trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
            trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
            if rnd_value <= 0.25:
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
            elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
                trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            elif rnd_value > 0.5 and rnd_value <= 0.75:
                trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        if self.rot:
                trfm_mat = np.dot(transforms3d.axangles.axangle2mat([0,1,0],angle), trfm_mat)
        for ptcloud in points:
            ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)

            if self.scale:
                ptcloud = ptcloud * scale
            result.append(ptcloud)

        return result[0],result[1]

    def __getitem__(self, index):
        partial_path = self.input_data[index]
        with h5py.File(partial_path, 'r') as f:
            partial = np.array(f['data'])

        if self.prefix == 'train' and self.sample:
            choice = np.random.permutation((partial.shape[0]))
            partial = partial[choice[:2048]]
            if partial.shape[0] < 2048:
                zeros = np.zeros((2048-partial.shape[0],3))
                partial = np.concatenate([partial,zeros])

        if self.prefix not in ["test"]:
            complete_path = partial_path.replace('partial','gt')
            with h5py.File(complete_path, 'r') as f:
                complete = np.array(f['data'])

            partial, complete = self.get_transform([partial, complete])

            complete = torch.from_numpy(complete)
            label = (self.labels[index])
            partial = torch.from_numpy(partial)

            return label, partial, complete
        else:
            partial = torch.from_numpy(partial)
            label = (self.labels[index])
            return label, partial, partial


class GeometricBreaksDataset:
    class BrokenPointsField:
        ''' Point Field.

        It provides the field to load point data. This is used for the points
        randomly sampled on the mesh, but separated in different files, following Geometric Breaks Dataset structure.

        Args:
            file_name (str): file name
            transform (list): list of transformations which will be applied to the points tensor
            multi_files (callable): number of files

        '''

        def __init__(self, file_name, unpackbits=False, multi_files=None, stddev=0.1):
            self.file_name = file_name
            self.file_base_name, self.file_extension = os.path.splitext(file_name)
            self.unpackbits = unpackbits
            self.multi_files = multi_files
            self.stddev = stddev

        def load(self, model_path, idx, category):
            ''' Loads the data point.

            Args:
                model_path (str): path to model
                idx (int): ID of data point
                category (int): index of category
            '''
            if self.multi_files is None:
                # file_path = os.path.join(model_path, self.file_name)
                broken_file_path = os.path.join(model_path, self.file_base_name + '_b' + self.file_extension)
                restoration_file_path = os.path.join(model_path, self.file_base_name + '_r' + self.file_extension)
                complete_file_path = os.path.join(model_path, self.file_base_name + '_c' + self.file_extension)
            else:
                if self.multi_files >= 0:
                    num = np.random.randint(self.multi_files)
                else:
                    # range may vary, so we choose from the folder
                    pattern = re.compile("^" + self.file_base_name + "_b_" + r"(\d+)\.npz")
                    files = set(
                        [file for file in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, file))])
                    # then get all the available numbers
                    nums = []
                    for file in files:
                        match = pattern.match(file)
                        if match is not None:
                            d = match.group(1)
                            # check if _r_ exists first
                            if self.file_base_name + f"_r_{d}.npz" in files:  # in keyword is fast in sets
                                nums.append(match.group(1))
                    num = np.random.choice(nums)

                # file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

                # model_{b, r}_{%d}.obj

                broken_file_path = os.path.join(model_path, self.file_base_name + f'_b_{num}' + self.file_extension)
                restoration_file_path = os.path.join(model_path,
                                                     self.file_base_name + f'_r_{num}' + self.file_extension)
                complete_file_path = os.path.join(model_path, self.file_base_name + '_c' + self.file_extension)

            try:
                broken_points_dict = np.load(broken_file_path)
                restoration_points_dict = np.load(restoration_file_path)
                complete_points_dict = np.load(complete_file_path)
            except:
                print("Error on loading something")
                print(broken_file_path, restoration_file_path, complete_file_path)
                raise

            # Break symmetry if given in float16:
            def break_symmetry(points):
                if points.dtype == np.float16:
                    points = points.astype(np.float32)
                    points += 1e-4 * np.random.randn(*points.shape)
                return points

            broken_points = break_symmetry(broken_points_dict['points'])
            restoration_points = break_symmetry(restoration_points_dict['points'])
            complete_points = break_symmetry(complete_points_dict['points'])

            return category, broken_points, restoration_points, complete_points

    def __init__(self, dataset_folder, prefix,
                 categories=None, no_except=True, transform=None, cfg=None):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
            cfg (yaml): config file
        '''
        # Get split
        splits = {
            'train': 'train',
            'val': 'val',
            'test': 'tests',
        }
        split = splits[prefix]
        dataset_folder = os.path.expanduser(dataset_folder)
        # Attributes
        self.dataset_folder = dataset_folder
        self.prefix = prefix
        self.field = self.BrokenPointsField('pointcloud.npz', unpackbits=False, multi_files=-1)
        self.no_except = no_except
        self.transform = transform
        self.cfg = cfg
        self.sample = 1

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]

        self.label_map = {}
        for i, c in enumerate(categories):
            self.label_map[c] = i
        self.label_map_inverse = categories

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            if split is None:
                self.models += [
                    {'category': c, 'model': m} for m in
                    [d for d in os.listdir(subpath) if (os.path.isdir(os.path.join(subpath, d)) and d != '')]
                ]

            else:
                split_file = os.path.join(subpath, split + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')

                if '' in models_c:
                    models_c.remove('')

                self.models += [
                    {'category': c, 'model': m}
                    for m in models_c
                ]

    def __len__(self):
        return len(self.models)

    @staticmethod
    def _infer_occ(noise, gt):
        # roughly estimate distance between points in point cloud
        dist_sq = cdist(gt, gt, 'sqeuclidean')
        np.fill_diagonal(dist_sq, np.inf)
        in_dist = np.max(np.min(dist_sq, axis=0))

        mask = (cdist(noise, gt, 'sqeuclidean').min(axis=1) <= in_dist)
        return mask

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        category = self.models[idx]['category']
        model = self.models[idx]['model']

        model_path = os.path.join(self.dataset_folder, category, model)

        try:
            label, partial, restoration, complete = self.field.load(model_path, idx, category)
        except Exception:
            if self.no_except:
                logger.warn(
                    'Error occured when loading field %s of model %s'
                    % ("BrokenPointsField", model)
                )
                return None
            else:
                raise

        # SubsampleAll
        partial = partial[np.random.permutation(partial.shape[0])][:2048]
        restoration = restoration[np.random.permutation(restoration.shape[0])][:2048]
        complete = complete[np.random.permutation(complete.shape[0])[:2048]]
        noise = np.random.uniform(-1 / 2, 1 / 2, size=(2048, partial.shape[1]))
        if self.prefix == 'test':
            occ = None
        else:
            occ = self._infer_occ(noise, complete)
            occ = torch.from_numpy(occ)

        partial = torch.from_numpy(partial)
        restoration = torch.from_numpy(restoration)
        complete = torch.from_numpy(complete)
        noise = torch.from_numpy(noise)

        if self.prefix == 'test':
            return label, partial, noise, complete, restoration, model
        else:
            return label, partial, noise, occ, restoration


if __name__ == '__main__':
    dataset = C3D_h5(prefix='test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=True, num_workers=0)
    for idx, data in enumerate(dataloader, 0):
        print(data.shape)




