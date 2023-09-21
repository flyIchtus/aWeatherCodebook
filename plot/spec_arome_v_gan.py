import spectrum_plots_simple as sps
from glob import glob

var_dict={'rr' : 0, 'u' : 1, 'v' : 2, 't2m' :3 , 'orog' : 4, 'z500': 5, 't850': 6, 'tpw850': 7}

filepath = "/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/spectrums/"
datapath = "/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"
output_dir = "/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/Spectral_plots/"
sample_list = glob(datapath+"_sample*")


lat=512 # 64 ; 128 ; 256 ; 512 ; 1024
bs=16 # 4 ; 8 ; 16 ; 32 ; 64
chm=2
lr=0.002
ckpt=24 # 147000 ; 588000 ; 294000 ; 72000
vars=['u','v','t2m'] #; 't850','tpw850' ; 'z500' ; 'u','v','t2m','z500','t850','tpw850' ; 'u','v','t2m','t850','tpw850'
rgb_level=3
N_samp=len(sample_list)
res=[256] # 128 ; 256
dom_size=256 # 128 ; 256
use_noise=True # True ; False
norma = 'classic'
zoom = False
no_mean = False
mean_pert = False

nb_res=len(res)
nb_var=len(vars)
program={0 :(1,N_samp)}

if __name__=="__main__":

    sps.spectral_plot(program=program, output_dir=output_dir, var_names=vars, lat=lat, bs=bs, chm=chm, zoom=zoom, no_mean=no_mean, lr=lr, 
        ckpt=ckpt, use_noise=use_noise, size=dom_size, N_samp=N_samp, rgb_level=rgb_level, res=res, norma=norma, mean_pert=mean_pert)