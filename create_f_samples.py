import torch
import numpy as np
#import metrics4arome as METR
import model.stylegan2 as RN
from multiprocessing import Pool
import os
from time import perf_counter
from collections import OrderedDict

"""
Created on Tue Apr 11 14:24:10 2023
@author: poulainl
""" 

def str2bool(v):
    return v.lower() in ('true')

def str2list(li):
    if type(li)==list:
        li2 = li
        return li2
    
    elif type(li)==str:
        li2=li[1:-1].split(',')
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))

def str2intlist(li):
    if type(li)==list:
        li2 = [int(p) for p in li]
        return li2
    
    elif type(li)==str:
        li2 = li[1:-1].split(',')
        li3 = [int(p) for p in li2]
        return li3
    
    else : 
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))

import argparse
parser = argparse.ArgumentParser(
        description="Fake samples creator"
    )
    
parser.add_argument(
        "--lat_dim", type=int, help="latent dimension to use",
        default=512
    )
parser.add_argument(
        "--bs", type=int, default=4, help="batch size"
    )
parser.add_argument(
        "--ckpt",
        type=int,
        default=24,
        help="which ckpt to use",
    )
parser.add_argument(
        "--ch_mul",
        type=int, 
        default=2,
        help="Channels multiplier",
    )

parser.add_argument(
        "--rgb",
        action="store_true",
        help="save the different outputs of the ToRGB layers",
)

parser.add_argument(
        "--no_mean",
        action="store_true",
        help="dataset with mean image taken off",
)

parser.add_argument(
        "--mean_pert",
        action="store_true",
        help="dataset split between mean and pert",
)

parser.add_argument(
        "--rgb_levels",
        type=str2intlist,
        default=[0,1,2,3],
        help="which keys of rgb dict to save. Refer to the stylegan.py code for indices reference"
)

parser.add_argument(
        "--var_names",
        type=str2list,
        default=['u','v','t2m'],
        help="list of variables to operate on"
)

parser.add_argument(
        "--use_noise",
        type=str2bool,
        default=True,
        help='whether the experiment was using noise_injection'
)

parser.add_argument(
        "--dom_size",
        type=int,
        default=256,
        help="size of the domain used"
)

parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
args = parser.parse_args()

lat = args.lat_dim
bs = args.bs
chm = args.ch_mul
lr = args.lr
ckpt = args.ckpt
size = args.dom_size

assert not (args.no_mean and args.mean_pert), "Can't have 0_mean and mean_pert activated at the same time"
no_mean = "_no_mean" if args.no_mean else ""
mean_pert = "_mean_pert" if args.mean_pert else ""


output_dir = "/scratch/mrmn/brochetc/GAN_2D/Exp_StyleGAN_final/Set_1/"+\
        f"stylegan2_stylegan_dom_{size}_lat-dim_{lat}_bs_{bs}_{lr}_{lr}_ch-mul_{chm}"
output_dir += f"_vars_{'_'.join(str(var_name) for var_name in args.var_names)}"
output_dir += f"_noise_{args.use_noise}{no_mean}{mean_pert}/Instance_14"

rgb_keys = {0:'prev_rgb',1:'prev_rgb_upsampled',2:'input_conved', 3:'current_rgb_out'}
    

ckpt_path = output_dir + '/models/'
ckpt_name = str(ckpt).zfill(6) + '.pt'
model_names =  RN.library['stylegan2']

device  = torch.device('cuda')
num_samples = 16384 # if not args.rgb else 65536

modelG_n = getattr(RN, model_names['G'])
modelG_ema = modelG_n(size=size, style_dim=lat, n_mlp=8, channel_multiplier=chm, 
                      nb_var=len(args.var_names)*2 if args.mean_pert else len(args.var_names), use_noise=args.use_noise)
modelG_ema = modelG_ema.to(device)

ckpt_dic = torch.load(ckpt_path + ckpt_name, map_location=device)['g_ema']

if 'module' in list(ckpt_dic.items())[0][0]: #juglling with Pytorch versioning and different module packaging
    ckpt_adapt = OrderedDict()
    for k in ckpt_dic.keys():
        k0 = k[7:]
        ckpt_adapt[k0] = ckpt_dic[k]
    modelG_ema.load_state_dict(ckpt_adapt)


else:
    modelG_ema.load_state_dict(ckpt)


modelG_ema.eval()
nb_batch = 128 if size==128 else 256
nb_batch = nb_batch*2 if args.rgb else nb_batch

t_s = perf_counter()
for j in range(nb_batch):
    if (j+1) % 16 == 0:
        print(f"batch {j+1}/{nb_batch}")
    z = torch.empty(num_samples//nb_batch, lat).normal_().to(device)
    with torch.no_grad():
        fake_samples, _, rgb = modelG_ema([z], return_rgb=True)
        if not args.rgb:
            np.save(output_dir + f"/samples/_Fsample_{int(ckpt)}_{j}.npy", fake_samples.cpu().numpy())
        if args.rgb:
            if not os.path.isdir(output_dir + "/toRGB_outs"):
                os.mkdir(output_dir + "/toRGB_outs")
            for i in args.rgb_levels:
                rgb_key = rgb_keys[i]
                savename = f"/toRGB_outs/RGBS_level_{i}_lat_{lat}_bs_{bs}_chm_{chm}_"\
                            + f"lr_{lr}_ckpt_{ckpt}_{j}.npy"
                np.save(output_dir + savename, rgb[rgb_key])    
print(f"{num_samples} images produced in {perf_counter()-t_s}s")
