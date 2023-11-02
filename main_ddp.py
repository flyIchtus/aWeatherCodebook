# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:43:33 2022

@author: brochetc
"""

import torch
import yaml
import os, sys
import memutils.memory_consumption as memco
from torch.distributed import init_process_group, destroy_process_group
from collections import OrderedDict

import model.cTrainer_ddp as trainer
from torch import distributed as dist

import plot.plotting_functions as plf

import metrics4arome as METR
from metrics4arome import sliced_wasserstein as SWD
import metrics4arome.wasserstein_distances as WD
import metrics4arome.spectrum_analysis as Spectral

from expe_init import get_expe_parameters
from distributed import (
    synchronize,
    is_main_gpu
)

sys.stdout.reconfigure(line_buffering=True, write_through=True)
try:
    local_rank = int(os.environ["LOCAL_RANK"])
except KeyError:
    local_rank = 0
torch.cuda.set_device(local_rank)
init_process_group(
    'nccl' if dist.is_nccl_available() else 'gloo',
    rank=local_rank,
    world_size=torch.cuda.device_count())


###############################################################################
############################# INITIALIZING EXPERIMENT #########################
###############################################################################

# config_file = os.getenv('CONFIG_FILE')#
# print("main_horovod CONFIG_FILE", config_file)

config = get_expe_parameters().parse_args()
if not os.path.exists(config.output_dir):
    os.mkdir(config.output_dir)
if not os.path.exists(config.output_dir + "/log"):
    os.mkdir(config.output_dir + "/log")
if not os.path.exists(config.output_dir + "/models"):
    os.mkdir(config.output_dir + "/models")
if not os.path.exists(config.output_dir + "/samples"):
    os.mkdir(config.output_dir + "/samples")

if config.conditional:
    from model.conditionalSockets import vqSocket

    VQparams_file = config.VQparams_file

    with open(VQparams_file) as f:
        VQparams = yaml.safe_load(f)['VQparams']

    Embedder = vqSocket(VQparams, discrete_level = config.discrete_level)
    Embedder.cuda()

    import model.cStylegan2 as RN

else:
    if config.model == 'resnet_aa':
        import model.resnets_antialiasing as RN

    elif config.model == 'resnet':
        import model.resnets as RN

    elif config.model == 'resnetV2':
        import model.resnetsV2 as RN

    elif config.model == 'stylegan2':
        import model.stylegan2 as RN

    elif config.model == 'stylegan2_fp16':
        import model.stylegan2_fp16 as RN

    elif config.model == 'swagan':
        import model.swagan as RN
    else:
        raise ValueError('Model unknown')

###############################################################################
############################ BUILDING MODELS ##################################
###############################################################################

load_optim = False

try:

    model_names = RN.library[config.model]

    modelG_n, modelD_n = getattr(RN, model_names['G']), getattr(RN, model_names['D'])

    if config.model == 'resnet_concat_bilin_emb' \
            or config.model == 'resnet_concat_rough_emb' \
            or config.model == 'cBN_proj':

        modelG = modelG_n(config.latent_dim, config.g_channels,
                          embed_scales=config.scales_G)
        modelD = modelD_n(config.d_channels, embed_scales=config.scales_D)

    elif config.train_type == 'stylegan':

        if config.model == 'stylegan2':

            modelG = modelG_n(config.crop_size[0], config.latent_dim, config.n_mlp, config.mlp_inj_level,
                              channel_multiplier=config.channel_multiplier, nb_var=len(config.var_names),
                              use_noise=config.use_noise)

            modelD = modelD_n(config.crop_size[0],
                              channel_multiplier=config.channel_multiplier, nb_var=len(config.var_names))

            modelG_ema = modelG_n(config.crop_size[0], config.latent_dim, config.n_mlp, config.mlp_inj_level,
                                  channel_multiplier=config.channel_multiplier, nb_var=len(config.var_names),
                                  use_noise=config.use_noise)
        elif config.model == 'stylegan2_fp16':

            modelG = modelG_n(config.crop_size[0], config.latent_dim, config.n_mlp,
                              channel_multiplier=config.channel_multiplier,
                              num_fp16_res=config.fp16_resolution)

            modelD = modelD_n(config.crop_size[0],
                              channel_multiplier=config.channel_multiplier,
                              num_fp16_res=config.fp16_resolution)

            modelG_ema = modelG_n(config.crop_size[0], config.latent_dim, config.n_mlp,
                                  channel_multiplier=config.channel_multiplier,
                                  num_fp16_res=config.fp16_resolution)


    else:

        modelG = modelG_n(config.latent_dim, config.g_channels)
        modelD = modelD_n(config.d_channels)

except KeyError:  # back to "default names", error-prone is not wished for!

    modelG = RN.ResNet_G(config.latent_dim, config.g_output_dim, config.g_channels)

    modelD = RN.ResNet_D(config.d_input_dim, config.d_channels)

if config.train_type == 'wgan-hinge' and config.sn_on_g:
    modelG.apply(RN.Add_Spectral_Norm)

if config.train_type == 'wgan-hinge':
    modelD.apply(RN.Add_Spectral_Norm)

if config.pretrained_model > 0:

    i = config.pretrained_model
    print(i, config.output_dir + f'/models/{str(i).zfill(6)}.pt')
    ckpt = torch.load(config.output_dir + f'/models/{str(i).zfill(6)}.pt')

    if 'module' in list(ckpt["g"].items())[0][0][:7]: #juglling with Pytorch versioning and different module packaging
        ckpt_adapt = OrderedDict()
        ckpt_adapt_ema = OrderedDict()
        for k,kema in zip(ckpt["g"].keys(), ckpt["g_ema"]):
            k0,k0ema = k[7:], kema[7:]
            ckpt_adapt[k0] = ckpt["g"][k]
            ckpt_adapt_ema[k0ema] = ckpt["g_ema"][k]

        modelG.load_state_dict(ckpt_adapt)
        modelG_ema.load_state_dict(ckpt_adapt_ema)
        modelG_ema.eval()

    else:
        modelG.load_state_dict(ckpt["g"])
        modelG_ema.load_state_dict(ckpt["g_ema"])
        modelG_ema.eval()

    if 'module' in list(ckpt["d"].items())[0][0][:7]: #juglling with Pytorch versioning and different module packaging
        ckpt_adapt = OrderedDict()
        for k in ckpt["d"].keys():
            k0 = k[7:]
            ckpt_adapt[k0] = ckpt["d"][k]
        modelD.load_state_dict(ckpt_adapt)


    else:
        modelD.load_state_dict(ckpt["d"])
else:

    ckpt = None

    modelG_ema.eval()

    trainer.accumulate(modelG_ema, modelG, 0)

    if config.ortho_init:
        modelG.apply(RN.Orthogonal_Init)

        modelD.apply(RN.Orthogonal_Init)

synchronize()

###############################################################################
######################### Defining metrics #############################
###############################################################################

# names used in test_metrics should belong to the metrics namespace --> on-the-fly definition of metrics

sliced_wd = SWD.SWD_API2(numpy=False, ch_per_ch=False)
setattr(METR, "SWD_metric_torch", 
                        METR.metric2D('Sliced Wasserstein Distance  ',\
                            sliced_wd.End2End,\
                            [str(var_name) for var_name in config.var_names], 
                            names=sliced_wd.get_metric_names(),
                            ))

setattr(METR, "spectral_dist_torch_"+"_".join(str(var_name) for var_name in config.var_names), 
                        METR.metric2D('Power Spectral Density RMSE', 
                            Spectral.PSD_compare_torch, 
                            [str(var_name) for var_name in config.var_names], 
                            names = [f"PSD{str(var[1:-1])}" for var in config.var_names],
                            ))

setattr(METR, "W1_center", 
                        METR.metric2D('Mean Wasserstein distance on center crop  ', 
                            WD.W1_center, 
                            [str(var_name) for var_name in config.var_names], 
                            names = ['W1_Center'],
                            ))

setattr(METR, "W1_Random", 
                        METR.metric2D('Mean Wasserstein distance on random selection  ', 
                            WD.W1_random, 
                            [str(var_name) for var_name in config.var_names], 
                            names = ['W1_random'],
                            ))


test_metr = ["W1_Random", "SWD_metric_torch"] # if not config.mean_pert else ["W1_Random"] # SWD won't work with mean_pert
#if not config.mean_pert:
test_metr = test_metr + ["spectral_dist_torch_"+"_".join(str(var_name) for var_name in config.var_names)] # same (or at least need some work)

ensemble_metr = ["spread_diff_torch", "mean_diff_torch"]

###############################################################################
######################### LOADING models and Data #############################
###############################################################################

print('creating trainer', flush=True)
TRAINER = trainer.Trainer(config,criterion="W1_center",\
                        test_metrics=test_metr, test_ensemble_metrics=ensemble_metr)


print('instantiating', flush=True)
modelG, modelD, modelG_ema, mem_g, mem_d, mem_opt, mem_cuda = TRAINER.instantiate(modelG, modelD, load_optim=ckpt, modelG_ema=modelG_ema)


#if is_main_gpu():
#    memco.log_mem_consumption(modelG, modelD, config, mem_g, mem_d, mem_opt, mem_cuda)
#    #print(f"Total number of parameters: {Gen_param} (Generator) + {Dis_param} (Discriminator) = {Gen_param + Dis_param}.\n")


###############################################################################
################################## TRAINING ###################################
##########################   (and online testing)  ############################
###############################################################################

TRAINER.fit_(modelG, modelD, Embedder, modelG_ema=modelG_ema)

###############################################################################
############################## Light POST-PROCESSING ##########################
############################ (of training output data) ########################

if is_main_gpu():
    plf.plot_metrics_from_csv(config.output_dir + '/log/', 'metrics.csv')

synchronize()

destroy_process_group()
