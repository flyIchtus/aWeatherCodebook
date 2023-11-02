import torch
import numpy as np
#import metrics4arome as METR
import model.cStylegan2 as RN
from multiprocessing import Pool
import os
from time import perf_counter
from collections import OrderedDict
from model.conditionalSockets import vqSocket
import data.utils as utils
import pandas as pd
import yaml
import copy

"""
Created on Tue Apr 11 14:24:10 2023
@author: brochetc on the basis of poulainl
"""

var_dict = {'rr' : 0, 'u' : 1, 'v' : 2 ,'t2m' : 3 }

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
        "--bs", type=int, default=32, help="batch size"
    )
parser.add_argument(
        "--ckpt",
        type=int,
        default=24,
        help="which ckpt to use",
    )

parser.add_argument(
    "--crop_indices",
    type=str2intlist,
    default=[78,206,55,183]
)
parser.add_argument(
        "--ch_mul",
        type=int, 
        default=1,
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
        default=128,
        help="size of the domain used"
)

parser.add_argument(
        "--real_data_dir",
        type=str,
        default='/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_done/',
        help="where to fetch conditional data"
)


parser.add_argument(
        "--VQparams_file",
        type=str,
        default='/home/mrmn/brochetc/aWeatherCodebook/configs/VQgan.yml',
        help="where to fetch conditional data"
)

parser.add_argument(
        "--dates_file",
        type=str,
        default='selected_inversion_dates_labels.csv',
        help="which dates to use for conditioning"
)

parser.add_argument(
        "--date_start",
        type=str,
        default='2020-06-15',
        help="starting date for conditioning"
    )
parser.add_argument(
        "--date_stop",
        type=str,
        default='2021-11-15',
        help="ending date for conditioning"
    )

parser.add_argument('--mean_file', type=str, default='mean_with_8_var.npy')
parser.add_argument('--max_file', type=str, default='max_with_8_var.npy')

parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
args = parser.parse_args()

args.Shape = (len(args.var_names), args.dom_size, args.dom_size)

lat = args.lat_dim
bs = args.bs
chm = args.ch_mul
lr = args.lr
ckpt = args.ckpt
size = args.dom_size


################ Loading dates reference ###################
df = pd.read_csv(args.real_data_dir + args.dates_file)

df_date = df.copy()

df_date['Date'] = pd.to_datetime(df_date['Date'])

df_extract = df_date[(df_date['Date']>=args.date_start) & (df_date['Date']<=args.date_stop)]

liste_dates = df_extract['Date'].unique()

var_indices = [var_dict[k] for k in args.var_names]

Means = np.load(f'{args.real_data_dir}{args.mean_file}')[var_indices].reshape(1,args.Shape[0],1,1)

Maxs = np.load(f'{args.real_data_dir}{args.max_file}')[var_indices].reshape(1,args.Shape[0],1,1)

scale = 1/0.95

output_dir = "/scratch/mrmn/brochetc/GAN_2D/Exp_Codebook/Set_2/"+\
        f"stylegan2_stylegan_dom_{size}_lat-dim_{lat}_bs_{bs}_{lr}_{lr}_ch-mul_{chm}"
output_dir += f"_vars_{'_'.join(str(var_name) for var_name in args.var_names)}"
output_dir += f"_noise_{args.use_noise}/Instance_2"

rgb_keys = {0:'prev_rgb',1:'prev_rgb_upsampled',2:'input_conved', 3:'current_rgb_out'}
    




ckpt_path = output_dir + '/models/'
ckpt_name = str(ckpt).zfill(6) + '.pt'
model_names =  RN.library['stylegan2']

device  = torch.device('cuda')
num_samples = 16384 # if not args.rgb else 65536



#################################### Loading generator
modelG_n = getattr(RN, model_names['G'])
modelG_ema = modelG_n(size=size, style_dim=lat, n_mlp=8, mlp_inj_level=1 , channel_multiplier=chm, 
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




############################# Loading embedder

VQparams_file = args.VQparams_file

with open(VQparams_file) as f:
    VQparams = yaml.safe_load(f)['VQparams']

Embedder = vqSocket(VQparams, discrete_level = 0)
Embedder.cuda()


lambda_t = 1.0
#########################################

nb_batch = 112


t_s = perf_counter()
for i,date in enumerate(liste_dates):
    for lt in range(8):
        datename = date.strftime('%Y-%m-%d')

        print(datename,lt)
        cond = utils.load_batch_from_timestamp(df_extract, date, lt, args.real_data_dir,
                                                 Shape=args.Shape, var_indices=var_indices,
                                                  crop_indices= args.crop_indices)
        cond0 = copy.deepcopy(cond)
        cond = utils.scale(cond, Means, Maxs, scale)
        #print(cond.mean(axis=(0,-2,-1)))
        #print(cond.std(axis=(0,-2,-1)))
        nb_cond = cond.shape[0]
        cond = torch.tensor(cond, dtype = torch.float32).repeat(nb_batch//nb_cond,1,1,1).to(device)
       
        z = torch.empty(nb_batch, lat).normal_().to(device)
        with torch.no_grad():
            y = Embedder(cond)
            fake_samples, _, _ = modelG_ema([5*z],y,lambda_t)
            
            fake_samples = fake_samples.cpu().numpy()
            fake_samples0 = utils.rescale(fake_samples, Means, Maxs, 0.95) 
            print("fake mean", fake_samples0.mean(axis=(0,-2,-1)))
            print("real mean",cond0.mean(axis=(0,-2,-1)))
            print("fake std", fake_samples0.std(axis=(0,-2,-1)))
            print("real std", cond0.std(axis=(0,-2,-1)))
            print("std rel diff", (fake_samples0.std(axis=(0,-2,-1)) - cond0.std(axis=(0,-2,-1))) / cond0.std(axis=(0,-2,-1)))

            np.save(output_dir + f"/samples/_Fsemble_{i}_{lt}.npy", fake_samples)
        
print(f"{num_samples} images produced in {perf_counter()-t_s}s")
