#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 13:54:10 2022

@author: brochetc

DataSet class from Importance_Sampled images

"""

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from tqdm import tqdm

def print_stat(image, inc="=", depht=2, type="numpy", tranpose=True):

    if type == "numpy":
        if tranpose:
            print("numpy tranpose image", image.shape)
            image = np.transpose(image, axes=[2, 1, 0])
            print("numpy tranpose image 2", image.shape)

        print("\n"+inc*depht +
              f" grid u  \t m {np.mean(image[0,:,:])},\t s {np.std(image[0,:,:])},\t min  {np.min(image[0,:,:])},\t max {np.max(image[0,:,:])}")
        print(inc*depht +
              f" grid v \t m {np.mean(image[1,:,:])},\t s {np.std(image[1,:,:])},\t min  {np.min(image[1,:,:])},\t max {np.max(image[1,:,:])}")
        print(inc*depht +
              f" grid t \t m {np.mean(image[2,:,:])},\t s {np.std(image[2,:,:])},\t min  {np.min(image[2,:,:])},\t max {np.max(image[2,:,:])}")
    else:
        if tranpose:
            image = torch.transpose(image, 0, 2)
        print("\n"+inc*depht +
              f" grid u  \t m {torch.mean(image[0,:,:])},\t s {torch.std(image[0,:,:])},\t min  {torch.min(image[0,:,:])},\t max {torch.max(image[0,:,:])}")
        print(inc*depht +
              f" grid v \t m {torch.mean(image[1,:,:])},\t s {torch.std(image[1,:,:])},\t min  {torch.min(image[1,:,:])},\t max {torch.max(image[1,:,:])}")
        print(inc*depht +
              f" grid t \t m {torch.mean(image[2,:,:])},\t s {torch.std(image[2,:,:])},\t min  {torch.min(image[2,:,:])},\t max {torch.max(image[2,:,:])}")


# reference dictionary to know what variables to sample where
# do not modify unless you know what you are doing

var_dict = {'rr': 0, 'u': 1, 'v': 2, 't2m': 3, 'orog': 4}

################


class ISDataset(Dataset):

    def __init__(self, data_dir, ID_file, var_indexes, crop_indexes,
                 add_coords=False, load_data_in_RAM=False, conv_data_in_float16=False):

        print("__init__ ISDataset")


        self.data_dir = data_dir
        self.labels = pd.read_csv(data_dir+ID_file)

        # portion of data to crop from (assumed fixed)

        self.CI = crop_indexes
        self.VI = var_indexes
        # self.coef_avg2D = coef_avg2D

        # adding 'positional encoding'
        self.add_coords = add_coords
        Means = np.load(data_dir+'mean_with_orog.npy')[self.VI]
        Maxs = np.load(data_dir+'max_with_orog.npy')[self.VI]
        self.means = list(tuple(Means))
        self.stds = list(tuple((1.0/0.95)*(Maxs)))
        self.load_data_in_RAM = load_data_in_RAM
        self.conv_data_in_float16 = conv_data_in_float16

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds),
                transforms.Normalize([-1, -1, -1], [2, 2, 2]),
            ]
        )

        # Load all the data in RAM
        # -------------------------

        # files = os.listdir(self.data_dir)
        # files_npy = [fichier for fichier in files if fichier.startswith(
        #     "_sample") and fichier.endswith(".npy")]


        if self.load_data_in_RAM : 
            self.samples = {}
            
            for idx, file_ in enumerate(self.labels.iloc[:, 0])  :
                sample_path = os.path.join(self.data_dir, file_)
                sample = np.float32(np.load(sample_path+'.npy')
                                    )[self.VI, self.CI[0]:self.CI[1], self.CI[2]:self.CI[3]]

                importance = self.labels.iloc[idx, 1]
                position = self.labels.iloc[idx, 2]
                # transpose to get off with transform.Normalize builtin transposition
                sample = sample.transpose((1, 2, 0))
                sample = self.transform(sample)
                if self.conv_data_in_float16 : 
                    sample = sample.half()
                self.samples[idx] = {}
                self.samples[idx]["sample"]= torch.transpose(sample, 0, 2)
                self.samples[idx]["importance"] = importance
                self.samples[idx]["position"] = position

                if idx % 5000 == 0 : 
                    print(f"\tidx : {idx}, gpu : {torch.cuda.current_device()}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load all the data in RAM 
        # ------------------------
        if self.load_data_in_RAM :
            return self.samples[idx]["sample"], self.samples[idx]["importance"], self.samples[idx]["position"]

        # Load the data for each sample
        # -----------------------------
        else :

            sample_path = os.path.join(self.data_dir, self.labels.iloc[idx, 0])
            sample = np.float32(np.load(sample_path+'.npy')
                                )[self.VI, self.CI[0]:self.CI[1], self.CI[2]:self.CI[3]]

            importance = self.labels.iloc[idx, 1]
            position = self.labels.iloc[idx, 2]

            # transpose to get off with transform.Normalize builtin transposition
            sample = sample.transpose((1, 2, 0))

            sample = self.transform(sample)

            if self.conv_data_in_float16 : 
                sample = sample.half()

            return torch.transpose(sample, 0, 2), importance, position


class ISData_Loader_train(ISDataset):
    def __init__(self,
                 batch_size=1,
                 var_indexes=[1, 2, 3],
                 crop_indexes=[78, 206, 55, 183],
                 path="data/train_IS_1_1.0_0_0_0_0_0_256_done_red/",
                 shuf=False,
                 add_coords=False,
                 num_workers=0, 
                 load_data_in_RAM=False, conv_data_in_float16=False):

        super().__init__(path, 'IS_method_labels.csv',
                         var_indexes, crop_indexes)

        print("__init__ ISData_Loader_train")

        self.path = path
        self.batch = batch_size
        if num_workers == 0:
            num_workers = self.batch*2
        self.shuf = shuf  # shuffle performed once per epoch
        self.VI = var_indexes
        self.CI = crop_indexes

        Means = np.load(path+'mean_with_orog.npy')[self.VI]
        Maxs = np.load(path+'max_with_orog.npy')[self.VI]

        self.means = list(tuple(Means))
        self.stds = list(tuple((1.0/0.95)*(Maxs)))
        self.add_coords = add_coords
        self.load_data_in_RAM = load_data_in_RAM
        self.conv_data_in_float16 = conv_data_in_float16


    def loader(self):

        print("-"*80)
        print("&"*80)
        print("loader TRAIN  load_data_in_RAM", self.load_data_in_RAM)
        print("loader TRAIN  conv_data_in_float16", self.conv_data_in_float16)
        print("&"*80)
        print("-"*80)

        dataset = ISDataset(self.path, 'IS_method_labels.csv',
                            self.VI, self.CI, 
                            load_data_in_RAM=self.load_data_in_RAM, 
                            conv_data_in_float16=self.conv_data_in_float16)

        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch,
                            num_workers=self.batch*2,
                            pin_memory=False,
                            shuffle=True,
                            drop_last=True,
                            )
        return loader, dataset

    def norm_info(self):
        return self.means, self.stds


class ISData_Loader_val(ISDataset):
    def __init__(self,
                 batch_size=1,
                 var_indexes=[1, 2, 3],
                 crop_indexes=[78, 206, 55, 183],
                 path="data/test_IS_1_1.0_0_0_0_0_0_256_done_red/",
                 shuf=False,
                 add_coords=False,
                 load_data_in_RAM=False, conv_data_in_float16=False):


        super().__init__(path, 'IS_method_labels.csv',
                         var_indexes, crop_indexes)

        self.path = path
        self.batch = batch_size
        self.shuf = shuf  # shuffle performed once per epoch
        self.VI = var_indexes
        self.CI = crop_indexes

        Means = np.load(path+'mean_with_orog.npy')[self.VI]
        Maxs = np.load(path+'max_with_orog.npy')[self.VI]

        self.means = list(tuple(Means))
        self.stds = list(tuple((1.0/0.95)*(Maxs)))
        self.add_coords = add_coords
        self.load_data_in_RAM = load_data_in_RAM
        self.conv_data_in_float16 = conv_data_in_float16

    def _prepare(self):


        print("-"*80)
        print("-"*80)
        print("_prepare VAL  load_data_in_RAM", self.load_data_in_RAM)
        print("_prepare VAL  conv_data_in_float16", self.conv_data_in_float16)
        print("-"*80)
        print("-"*80)

        ISDataset(self.path, 'IS_method_labels.csv',
                  self.VI, self.CI,
                  load_data_in_RAM = self.load_data_in_RAM,
                  conv_data_in_float16=self.conv_data_in_float16)

    def loader(self):


        print("-"*80)
        print("-"*80)
        print("loader VAL  load_data_in_RAM", self.load_data_in_RAM)
        print("loader VAL  conv_data_in_float16", self.conv_data_in_float16)
        print("-"*80)
        print("-"*80)

        dataset = ISDataset(self.path, 'IS_method_labels.csv',
                            self.VI, self.CI,
                            load_data_in_RAM=self.load_data_in_RAM,
                            conv_data_in_float16=self.conv_data_in_float16)

        loader = DataLoader(dataset=dataset,
                            batch_size=self.batch,
                            num_workers=1,
                            pin_memory=False,
                            shuffle=True,
                            drop_last=True,
                            )
        return loader, dataset

    def norm_info(self):
        return self.means, self.stds

