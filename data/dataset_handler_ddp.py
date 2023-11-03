#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:44:08 2022

@authors: gandonb, rabaultj, brochetc


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
from data.statsMasked import normalizeUnderMask
from data.normalize_funcs import MultiOptionNormalize
import random
import yaml

#random.seed(0)
################ reference dictionary to know what variables to sample where
################ do not modify unless you know what you are doing 

var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4, 'z500': 5, 't850': 6, 'tpw850': 7}


################
class ISDataset(Dataset):

    def __init__(self, config, dataset_handler_yaml, sample_method, variable_indices,
     transform):
        self.config = config
        self.dataset_handler_yaml = dataset_handler_yaml
        self.sample_method = sample_method
        self.VI = variable_indices
        self.transform = transform
        self.labels = pd.read_csv(f"{self.config.data_dir}{self.config.id_file_train}")
        
        ## choosing the sampling method (either random crop or fixed coordinates)

        assert sample_method in ['random', 'coords']
        if sample_method=='coords' :
            self.CI = self.config.crop_indexes
            try:
                assert self.config.crop_indexes is not None
            except AssertionError:
                raise ValueError(f"crop_indexes are {self.CI} and sample_method is coordinates")
            try:
                assert self.CI[1] - self.CI[0] == self.config.crop_size[0]
                assert self.CI[3] - self.CI[2] == self.config.crop_size[1]
            except AssertionError :
                raise ValueError(f"Provided crop indexes ({self.CI}) should match crop size ({self.config.crop_size})")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        

        ####### finding main (conditioning) sample
        cond_path = os.path.join(self.config.data_dir, self.labels['Name'].iloc[idx])
        cond_name = self.labels['Name'].iloc[idx]
        if self.sample_method == 'coords':
            crop_X0 = self.CI[0]
            crop_X1 = self.CI[1]
            crop_Y0 = self.CI[2]
            crop_Y1 = self.CI[3]
        if self.sample_method == 'random':
            crop_X0 = np.random.randint(0, high=self.full_size[0] - self.crop_size[0])
            crop_X1 = crop_X0 + self.crop_size[0]
            crop_Y0 = np.random.randint(0, high=self.full_size[1] - self.crop_size[1])
            crop_Y1 = crop_Y0 + self.crop_size[1]

        cond = np.float32(np.load(cond_path + '.npy')) \
                [self.VI, crop_X0:crop_X1, crop_Y0:crop_Y1]

        try:
            importance = self.labels['Importance'].iloc[idx]
            position = self.labels['Position'].iloc[idx]
        
        except KeyError:
            pass

        #### finding target sample
        cond_member = self.labels['Member'].iloc[idx] #member
        cond_date = self.labels['Date'].iloc[idx] #date
        cond_lt = self.labels['Leadtime'].iloc[idx] #leadtime
        other_members = [i for i in range(16) if i!=cond_member]
        mb = random.sample(other_members,1)[0]

        #print(cond_member, mb)
        target_df = self.labels[(self.labels['Date']==str(cond_date)) & (self.labels['Leadtime']==int(cond_lt)) & (self.labels['Member']==int(mb))]

        target_path = os.path.join(self.config.data_dir, 
                        target_df['Name'].values[0])
        target = np.float32(np.load(target_path + '.npy')) \
                [self.VI, crop_X0:crop_X1, crop_Y0:crop_Y1]

        if 'rr' in self.config.var_names: #applying transformations on rr only if selected
            for _ in range(self.dataset_handler_yaml["rr_transform"]["log_transform_iteration"]):
                target[0], cond[0] = np.log(1 + target[0]), np.log(1 + cond[0])
            if self.dataset_handler_yaml["rr_transform"]["symetrization"] and np.random.random() <= 0.5:
                target[0], cond[0] = -target[0], -cond[0]
                
        cond = cond.transpose((1, 2, 0))
        target = target.transpose((1,2,0))

        cond = self.transform(cond)
        target = self.transform(target)

        return target, cond

class ISData_Loader():

    def __init__(self, dataset_type, config, shuf=False):
        print(f"{dataset_type} files data loader...")
        self.config = config
        if dataset_type == "Train":
            self.batch_size = self.config.batch_size
        else:
            self.batch_size = self.config.test_samples
        self.VI = [var_dict[var] for var in self.config.var_names]
        self.sampled_indices = {var: i for i, var in enumerate(self.config.var_names)} # corresponding indices in the prepared data (after sampling)

        self.shuf = shuf #shuffle performed once per epoch

        self.dataset_handler_yaml = self.read_dataset_handler_config_file()
        self.maxs, self.mins, self.means, self.stds = self.init_normalization()

        if self.stds is not None:
            self.stds *= 1.0 / 0.95

    def read_dataset_handler_config_file(self):
        print(f"{self.config.config_dir}{self.config.dataset_handler_config}")
        with open(f"{self.config.config_dir}{self.config.dataset_handler_config}", "r") as dataset_handler_config_file:
            print(f"{self.config.config_dir}{self.config.dataset_handler_config} opened...")
            return yaml.safe_load(dataset_handler_config_file)

    def init_normalization(self):
        normalization_type = self.dataset_handler_yaml["normalization"]["type"]
        if normalization_type == "mean":
            means, stds = self.load_stat_files(normalization_type, "mean", "std")
            return None, None, means[self.VI], stds[self.VI]
        if normalization_type == "minmax":
            maxs, mins = self.load_stat_files(normalization_type, "max", "min")
            return maxs[self.VI], mins[self.VI], None, None
        print("No normalization set")
        return None, None, None, None

    def load_stat_files(self, normalization_type, str1, str2):
        mean_or_max_filename = f"{str1}_{self.dataset_handler_yaml['stat_version']}"
        mean_or_max_filename += "_log" * self.dataset_handler_yaml["rr_transform"]["log_transform_iteration"]
        std_or_min_filename = f"{str2}_{self.dataset_handler_yaml['stat_version']}"
        std_or_min_filename += "_log" * self.dataset_handler_yaml["rr_transform"]["log_transform_iteration"]
        if self.dataset_handler_yaml["normalization"]["per_pixel"]:
            mean_or_max_filename += "_ppx"
            std_or_min_filename += "_ppx"
        mean_or_max_filename += ".npy"
        std_or_min_filename += ".npy"
        print(f"Normalization set to {normalization_type}")
        means_or_maxs = np.load(f"{self.config.data_dir}{self.dataset_handler_yaml['stat_folder']}{mean_or_max_filename}").astype('float32')
        print(f"{str1} file found")
        stds_or_mins = np.load(f"{self.config.data_dir}{self.dataset_handler_yaml['stat_folder']}{std_or_min_filename}").astype('float32')
        print(f"{str2} file found")
        return means_or_maxs, stds_or_mins

    def transform(self):
        options = [ToTensor()]
        normalization = self.dataset_handler_yaml["normalization"]["type"]
        if normalization != "None":
            if 'rr' in self.config.var_names and self.dataset_handler_yaml["rr_transform"]["symetrization"]: #applying transformations on rr only if selected
                if normalization == "means":
                    self.means[0] = np.zeros_like(self.means[0])
                elif normalization == "minmax":
                    self.mins[0] = -self.maxs[0]
        options.append(MultiOptionNormalize(self.means, self.stds, self.maxs, self.mins, self.config, self.dataset_handler_yaml))
        transform = Compose(options)
        return transform

    def loader(self, world_size=None, local_rank=None, kwargs=None):

        if kwargs is not None:
            with FileLock(os.path.expanduser("~/.horovod_lock")):  # if absent, causes SIGSEGV error

                if self.config.crop_indexes is not None :
                    sample_method = 'coords'
                else:
                    sample_method = 'random'
                dataset = ISDataset(self.config, self.dataset_handler_yaml, sample_method, self.VI, self.transform())

        self.sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank)
        if kwargs is not None:
            loader = DataLoader(dataset = dataset,
                            batch_size = self.batch_size,
                            shuffle = self.shuf,
                            sampler = self.sampler,
                            drop_last = True,
                                num_workers=1,
                            **kwargs)
        else:
            loader = DataLoader(dataset = dataset,
                            batch_size = self.batch_size,
                            shuffle = self.shuf,
                            sampler = self.sampler,
                            drop_last = True,
                            num_workers=1)
        return loader
