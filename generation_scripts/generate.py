import argparse

import torch
from torchvision import utils
from model.stylegan2 import Generator
from numpy import save
import horovod.torch as hvd

hvd.init()

def str2list(li):
    if type(li)==list:
        li2 = li
        return li2
    
    elif type(li)==str:
        li2=li[1:-1].split(',')
        return li2
    
    else:
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))

def generate(args, g_ema, mean_latent, step):
    print(step)
    with torch.no_grad():
        g_ema.eval()
        for t in range(args.n_batches) :
            sample_z = torch.randn(args.sample, args.latent).cuda()
            sample, _ = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent
	                   )
            save(args.output_dir+'_Fsample_'+str(step)+'_'+str(t)+'.npy', sample.detach().cpu().numpy())


if __name__ == "__main__":

    torch.cuda.set_device(hvd.local_rank())

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=128, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=256,
        help="number of samples to be generated per batch",
    )
    
    parser.add_argument(
        "--n_batches", type=int, default=1, help="number of batches to be generated"
    )
    
    parser.add_argument(
        "--list_steps", type=str2list, default=[141000], help="list of training steps to be used as checkpoints"
    )

    parser.add_argument(
        "--output_dir", type=str, default="/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/Set_1/stylegan2_stylegan_512_32_0.002_0.002/Instance_3/samples/" # change with your path
    )

    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/Set_1/stylegan2_stylegan_512_32_0.002_0.002/Instance_3/models/", # change with your path
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).cuda()
    
    for step in args.list_steps :
        checkpoint = torch.load(args.ckpt+f'{str(step).zfill(6)}.pt')
        g_ema.load_state_dict(checkpoint["g_ema"])
        if args.truncation < 1:
            with torch.no_grad():
                mean_latent = g_ema.mean_latent(args.truncation_mean)
        else:
            mean_latent = None

        generate(args, g_ema, mean_latent, step)
