#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:44:08 2022

@author: brochetc


DataSet/DataLoader classes from Importance_Sampled images
DataSet:DataLoader classes for test samples

"""

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor, Normalize, Compose
from filelock import FileLock
from multiprocessing import Manager

################ reference dictionary to know what variables to sample where
################ do not modify unless you know what you are doing 

var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4, 'z500': 5, 't850': 6, 'tpw850': 7}


class DatasetCache(object):
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.manager = Manager()
        self._dict = self.manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError('Data caching is disabled and get function is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, sample, importance, pos):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (sample, importance, pos)


################
class ISDataset(Dataset):

    def __init__(self, data_dir, ID_file, var_indices, crop_size, full_size, \
                 transform=None,
                 sample_method='random',
                 crop_indexes=None,
                 use_cache=False):

        self.data_dir = data_dir
        self.transform = transform
        self.cache = DatasetCache(use_cache=use_cache)
        if use_cache:
            import resource
            resource.setrlimit(
                resource.RLIMIT_CORE,
                (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
            
        self.labels = pd.read_csv(data_dir + ID_file)
        self.full_size = full_size  # size of the full data samples

        ## portion of data to crop from (assumed fixed)

        self.crop_size = crop_size  # shoud be a tuple of integers (size H, size W)
        
        #print("self.crop_size", self.crop_size)

        self.VI = var_indices

        ## choosing the sampling method (either random crop or fixed coordinates)

        assert sample_method in ['random', 'coords']

        #print(sample_method)

        self.sample_method = sample_method

        if sample_method == 'coords':

            try:
                assert crop_indexes is not None

            except AssertionError:

                raise ValueError('crop_indexes is None and sample_method \
                                 is coordinates')

            self.CI = crop_indexes

            try:
                assert self.CI[1] - self.CI[0] == self.crop_size[0]

                assert self.CI[3] - self.CI[2] == self.crop_size[1]

            except AssertionError:

                raise ValueError(f'Provided crop indexes ({self.CI}) should match \
                                 crop size ({self.crop_size})')

    def reset_memory(self):
        self.cache.reset()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.cache.is_cached(idx):
            sample, importance, position = self.cache.get(idx)
        else:

            ####### finding main (conditioning) sample
            cond_path = os.path.join(self.data_dir, self.labels.iloc[idx, 0])
            cond_name = self.labels.iloc[idx, 0]
            if self.sample_method == 'coords':
                crop_X0 = CI[0]
                crop_X1 = CI[1]
                crop_Y0 = CI[2]
                crop_Y1 = CI[3]
            if self.sample_method == 'random':
                crop_X0 = np.random.randint(0, high=self.full_size[0] - self.crop_size[0])
                crop_X1 = crop_X0 + self.crop_size[0]
                crop_Y0 = np.random.randint(0, high=self.full_size[1] - self.crop_size[1])
                crop_Y1 = crop_Y0 + self.crop_size[1]

            cond = np.float32(np.load(cond_path + '.npy')) \
                    [self.VI, crop_X0:crop_X1, crop_Y0:crop_Y1]

            importance = self.labels.iloc[idx, 1]
            position = self.labels.iloc[idx, 2]

            #### finding target sample
            cond_member = self.labels.loc[cond_name,'Member']
            cond_date = self.labels.loc[cond_name,'Date']
            cond_lt = self.labels.loc[cond_name,'LeadTime']
            other_members = [i for i in range(16) if i!=cond_member]
            mb = random.sample(other_members,1)

            target_path = os.path.join(self.data_dir, 
                            self.labels.loc[(self.labels['Date']==cond_date)\
                             & (self.labels['LeadTime']==cond_lt) \
                             & (self.labels['Member']=mb)]['Name'])

            target = np.float32(np.load(target_path + '.npy'))
                    [self.VI, crop_X0:crop_X1, crop_Y0:crop_Y1]

            cond = cond.transpose((1, 2, 0))
            target = target.transpose((1,2,0))

            if self.transform:
                cond = self.transform(cond)
                target = self.transform(target)


            self.cache.cache(idx, target, cond, importance, position)

        return target, cond, importance, position


class ISData_Loader():

    def __init__(self, path, batch_size, variables, crop_size, full_size,
                 crop_indexes=None, \
                 shuf=False,
                 mean_file='mean_with_8_var.npy',
                 max_file='max_with_8_var.npy',
                 id_file='IS_method_labels_8_var.csv',
                 use_cache=False):

        self.path = path
        self.batch = batch_size
        self.use_cache = use_cache

        self.shuf = shuf  # shuffle performed once per epoch

        self.crop_size = crop_size
        self.full_size = full_size

        self.variables = variables

        self.VI = [var_dict[var] for var in variables]

        self.sampled_indices = {var: i for i, var in enumerate(
            self.variables)}  # corresponding indices in the prepared data (after sampling)

        self.CI = crop_indexes

        Means = np.load(path + mean_file)[self.VI]
        Maxs = np.load(path + max_file)[self.VI]

        self.means = list(tuple(Means))
        self.stds = list(tuple((1.0 / 0.95) * (Maxs)))
        self.id_file = id_file

    def transform(self, totensor, normalize):

        options = []
        if totensor:
            options.append(ToTensor())

        if normalize:
            options.append(Normalize(self.means, self.stds))

        transform = Compose(options)
        return transform

    def loader(self, hvd_size=None, hvd_rank=None, kwargs=None):

        if kwargs is not None:
            with FileLock(os.path.expanduser("~/.horovod_lock")):  # if absent, causes SIGSEGV error

                if self.CI is not None:
                    sample_method = 'coords'
                else:
                    sample_method = 'random'

                dataset = ISDataset(self.path, self.id_file,  ## CHANGED HERE
                                    self.VI,
                                    self.crop_size,
                                    self.full_size,
                                    self.transform(True, True),  # remettre True
                                    sample_method=sample_method,
                                    crop_indexes=self.CI,
                                    use_cache=self.use_cache)  # coordinates system

        self.sampler = DistributedSampler(
            dataset, num_replicas=hvd_size, rank=hvd_rank
        )
        if kwargs is not None:

            loader = DataLoader(dataset=dataset,
                                batch_size=self.batch,
                                shuffle=self.shuf,
                                sampler=self.sampler,
                                **kwargs
                                )
        else:
            loader = DataLoader(dataset=dataset,
                                batch_size=self.batch,
                                shuffle=self.shuf,
                                sampler=self.sampler,
                                )
        return loader
