#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:42:42 2022

@author: brochetc


General metrics

TODO : implement structure functions

"""

import torch
import numpy as np



############################ General simple metrics ###########################

def intra_map_var(useless_data, data, usetorch=True):
    if usetorch :
        res=torch.mean(torch.var(data, dim=(2,3)), dim=0)
    else :
        res=np.mean(np.var(data, axis=(2,3)), axis=0)
    return res

def inter_map_var(useless_data,data, usetorch=True):
    if usetorch :
        res=torch.mean(torch.var(data, dim=0), dim=(1,2))
    else :
        res=np.mean(np.var(data,axis=0), axis=(1,2))
    return res

def orography_RMSE(fake_batch, test_data, usetorch=False):
    orog=test_data[0,-1:,:,:]
    fake_orog=fake_batch[:,-1:,:,:]
    if usetorch :
        res=torch.sqrt(((fake_orog-orog)**2).mean())
    else :
        res=np.sqrt(((fake_orog-orog)**2).mean())
    return res

def spread_diff(real, fake, maxs=None, scale=1.0/0.95, usetorch=True):

    if usetorch:

        maxs = torch.tensor(maxs)

        diff = torch.std(real, dim=0) - torch.std(fake, dim=0) #delta degree of freedom is 1 by default in torch

        diff = scale * maxs.to(diff.device) * diff.mean(dim=(-2,-1))

    else :

        maxs = np.array(maxs)

        diff = np.std(real, axis=0, ddof=1) - np.std(fake, axis=0, ddof=1) # delta degree of freedom is 0 by default in numpy

        diff = scale * maxs * diff.mean(axis=(-2,-1))

    return diff

def mean_diff(real, fake, maxs=None, scale=1.0/0.95, usetorch=True):

    if usetorch:

        maxs = torch.tensor(maxs)

        diff = torch.mean(real, dim=0) - torch.mean(fake, dim=0)

        diff = scale * maxs.to(diff.device) * diff.mean(dim=(-2,-1))

    else :

        maxs = np.array(maxs)

        diff = np.mean(real, axis=0) - np.mean(fake, axis=0) 

        diff = scale * maxs * diff.mean(axis=(-2,-1))

    return diff

