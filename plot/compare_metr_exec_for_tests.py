#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:27:26 2023

@author: poulainauzeaul
"""

import plotting_functions as plf
import argparse

def launch_compare(type_):
    if type_=='lat':
        lat_dims = [64,128,256,512,1024]
        chms = [2]
        bss = [4]
        steps = [list(range(25))]#[51000,102000,147000],
                 #[51000,102000,147000],
                 #[51000,102000,147000],
                 #[51000,102000,147000],
                 #[51000,102000,147000]]
        set_path = "/scratch/mrmn/brochetc/GAN_2D/Exp_StyleGAN_final/"
        plf.compare_stats_metrics(set_path=set_path, steps=steps, 
                                add_name= [str(i) for i in range(25)].join('_')+ '_16384', lat_dims=lat_dims,
                                chms=chms, bss=bss, x_label='steps', dom_size=256)

    if type_=='bs':
        lat_dims = [512]
        chms = [2]
        bss = [8,16,32,64]
        steps = [[102000,204000,408000,588000],
                [51000,102000,204000,294000],
                [24000,51000,102000,147000],
                [12000,24000,51000,75000]]
        set_path = "/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/"
        plf.compare_stats_metrics(set_path=set_path, steps=steps, 
                                add_name='_16384_16384', lat_dims=lat_dims,
                                chms=chms, bss=bss, x_label='images')

    if type_=='noise':
        lat_dims = [512]
        chms = [2]
        bss = [16]
        steps = [[51000,102000,147000],
                 [51000,102000,147000]]
        use_noises = [False, True]
        set_path = "/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/"
        plf.compare_stats_metrics(set_path=set_path, steps=steps, 
                                add_name='_16384_16384', lat_dims=lat_dims,
                                chms=chms, bss=bss, x_label='steps', use_noises=use_noises)

    if type_=='var':
        lat_dims = [512]
        chms = [2]
        bss = [16]
        steps = [[51000,102000,147000],
                 [51000,102000,147000],
                 [51000,102000,147000],
                 [51000,102000,147000],
                 [51000,102000,147000]]
        varss=[['t2m'],['u','v'],['u','v','t2m'],
            ['u','v','t2m','t850','tpw850'],
            ['u','v','t2m','z500','t850','tpw850']]
        set_path = "/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/"
        plf.compare_stats_metrics(set_path=set_path, steps=steps, 
                                add_name='_16384_16384', lat_dims=lat_dims,
                                chms=chms, bss=bss, varss=varss, 
                                x_label='steps')

    if type_=='domain':
        lat_dims = [512]
        chms = [2]
        bss = [16]
        steps = [[51000,102000,147000],
                [51000,102000,147000]
                ]
        varss=[['u','v','t2m']]
        set_path = "/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/"
        plf.compare_stats_metrics(set_path=set_path, steps=steps, 
                                add_name='_16384_16384', lat_dims=lat_dims,
                                chms=chms, bss=bss, varss=varss, 
                                x_label='steps')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="gloubiboulga", help="which test to run")
    args = parser.parse_args()
    launch_compare(type_=str(args.type))