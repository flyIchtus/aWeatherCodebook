#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:59:24 2023

@author: brochetc, poulainauzeaul
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import os
from glob import glob
from metrics4arome import spectrum_analysis as sa
from plotting_functions import compute_normed_rgb, multiline_label
mpl.rcParams['axes.linewidth'] = 2.5
var_dict={'rr' : 0, 'u' : 1, 'v' : 2, 't2m' :3 , 'orog' : 4, 'z500': 5, 't850': 6, 'tpw850': 7}

filepath = "/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/spectrums/"
datapath = "/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"



# chnager norma pour stds/means/0.95
def create_normalized_spectrum_arome(size=128, norma='classic', no_mean=False):
    vars = list(var_dict.keys())

    datapath = "/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/IS_1_1.0_0_0_0_0_0_256_mean_0/" if no_mean else datapath
    means_fname = 'mean_with_8_var.npy' if not no_mean else "mean_with_0_mean.npy"
    maxs_fname = 'max_with_8_var.npy' if not no_mean else "max_with_0_mean.npy"

    Means = np.load(datapath + means_fname)[[var_dict[var] for var in vars]].reshape(len(vars),1,1)
    Maxs = np.load(datapath + maxs_fname)[[var_dict[var] for var in vars]].reshape(len(vars),1,1)        

    Stds = (1.0/0.95) * Maxs
    
    size_str = f"{size}x{size}"
    crop_ind = [78,206,55,183] if size==128 else [0,256,0,256]
    if norma.lower()!='classic':
        norma='std'
    spec_arome_norma_name = filepath + f"spec_arome_norm_{norma}_{'_'.join(var for var in vars)}"+\
                          f"_{size_str}"
    spec_arome_norma_name += "_no_mean.npy" if no_mean else ".npy"
    if not os.path.isfile(spec_arome_norma_name):
        print("Creating Arome spectrum")
        arome_norma_name = filepath + f"arome_{'_'.join(var for var in vars)}"+\
                          f"_{size_str}_normalised_{norma}"
        arome_norma_name += "_no_mean.npy" if no_mean else ".npy"
        if not os.path.isfile(arome_norma_name):
            print(f"Creating the normalised ({norma}) arome npy")
            sample_list = glob(datapath+"_sample*")
            arr = np.zeros((len(sample_list),len(vars),size,size))
            for i in range(len(sample_list)):
                if i%10000==0:
                    print(i)
                arr[i] = np.load(sample_list[i])[[var_dict[var] for var in vars],
                                                 crop_ind[0]:crop_ind[1],crop_ind[2]:crop_ind[3]]
            
            if norma.lower()=='classic':
                arr = (arr-Means)/Stds
            else:
                std = np.std(arr, axis=(0,-1,-2))
                arr = arr/std.reshape(1,len(vars),1,1)
            np.save(arome_norma_name, arr)
            print("Data created and saved")
        else:
            arr = np.load(arome_norma_name)
        spec_arome = sa.PowerSpectralDensity(arr)
        np.save(spec_arome_norma_name, spec_arome)
        print("Spectrum created !")
    else:
        print("Arome spectrum already created")



def create_normalized_spectrum_gan(program, vars=['u','v','t2m'], lat=512, bs=16, chm=2, lr=0.002, ckpt=147000, 
        use_noise=True, size=128, N_samp=65536, rgb_level=3, res=[128], norma='classic', no_mean=False, mean_pert=False):

    assert not (no_mean and mean_pert), "You can't have no_mean and mean_pert activated at the same time"
    # classic norma: with Means and Stds, otherwise with spatial std
    
    fpath = "/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/"
    rgb_path = f'/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/Finished_experiments/'+\
        f"stylegan2_stylegan_dom_{size}_lat-dim_{lat}_bs_{bs}_{lr}_{lr}_ch-mul_{chm}"+\
        f"_vars_{'_'.join(var for var in vars)}_noise_{use_noise}" +\
        ("_no_mean" if no_mean else "") + ("_mean_pert" if mean_pert else "") +\
        "/Instance_1/toRGB_outs/"
    
    size_str = f"{size}x{size}"
    nb_res = len(res)
    nb_var = len(vars)
    if norma.lower()!='classic':
        norma='std'
    name = f"rgb_lvl_{rgb_level}_{size_str}_{lat}_{bs}_{chm}_{lr}_{'_'.join(var for var in vars)}_{use_noise}"+\
            f"_res_{'_'.join(str(r) for r in res)}_norma_{norma}_{ckpt}" + ("_no_mean" if no_mean else "")
    name += "_mean_pert" if mean_pert else ""
    if not os.path.isfile(filepath + f"spec_gan_{name}.npy"):
        print("Creating GAN spectrum")
        spectrum_f = np.zeros((nb_res,nb_var,45*(size//128)))
        for i, r in enumerate(res):
            name2 = f"rgb_lvl_{rgb_level}_{size_str}_{lat}_{bs}_{chm}_{lr}_{'_'.join(var for var in vars)}_{use_noise}"+\
            f"_res_{r}_norma_{norma}_N_{N_samp}_{ckpt}" + ("_no_mean" if no_mean else "")
            name2 += "_mean_pert" if mean_pert else ""

            if not os.path.isfile(rgb_path + f"{name2}.npy"):
                print(f"Creating the normalised ({norma}) GAN npy")
                compute_normed_rgb(program=program, lat=lat, bs=bs, chm=chm, lr=lr, ckpt=ckpt, vars=vars, no_mean=no_mean,
                        rgb_level=rgb_level, use_noise=use_noise, fpath=fpath, norma=norma, dom_size=size, mean_pert=mean_pert)
                print("Data created and saved")
            arr = np.load(rgb_path + f"{name2}.npy")
            spectrum = sa.PowerSpectralDensity(arr)
            spectrum_f[i] = spectrum
        np.save(filepath + f"spec_gan_{name}.npy", spectrum_f)
        print("Spectrum created !")
    else:
        print("GAN spectrum already created !")



def load_arome_spectrum(var_names=['u','v','t2m'], size=128, norma='classic', no_mean=False):
    size_str = f"{size}x{size}"
    vars = list(var_dict.keys())
    if norma.lower()!='classic':
        norma='std'
    spec_arome_norma_name = filepath + f"spec_arome_norm_{norma}_{'_'.join(var for var in vars)}"+\
                          f"_{size_str}"
    spec_arome_norma_name += "_no_mean.npy" if no_mean else ".npy"
    if not os.path.isfile(spec_arome_norma_name):
        create_normalized_spectrum_arome(size=size, norma=norma, no_mean=no_mean)
    arr = np.load(spec_arome_norma_name)
    return arr[[var_dict[var] for var in var_names]]



def load_gan_spectrum(program, vars=['u','v','t2m'], lat=512, bs=16, chm=2, lr=0.002, ckpt=147000, 
        use_noise=True, size=128, N_samp=65536, rgb_level=3, res=[128], norma='classic', no_mean=False, mean_pert=False):

    assert not (no_mean and mean_pert), "You can't have no_mean and mean_pert activated at the same time"
    size_str = f"{size}x{size}"
    name = f"rgb_lvl_{rgb_level}_{size_str}_{lat}_{bs}_{chm}_{lr}_{'_'.join(var for var in vars)}_{use_noise}"+\
            f"_res_{'_'.join(str(r) for r in res)}_norma_{norma}_{ckpt}" + ("_no_mean" if no_mean else "")
    name += "_mean_pert" if mean_pert else ""
    
    if not os.path.isfile(filepath + f"spec_gan_{name}.npy"):
        create_normalized_spectrum_gan(program=program, vars=vars, lat=lat, bs=bs, chm=chm, lr=lr, ckpt=ckpt, use_noise=use_noise,
            size=size, N_samp=N_samp, rgb_level=rgb_level, res=res, norma=norma, no_mean=no_mean, mean_pert=mean_pert)
    arr = np.load(filepath + f"spec_gan_{name}.npy")
    return arr



def plot_arome_spectrum(var_names=['u','v','t2m'], size=128, norma="classic", no_mean=False):
    
    var_names = sorted(var_names, key=lambda var:var_dict[var])
    N_var = len(var_names)
    
    spectrum_arome = load_arome_spectrum(var_names, size, norma=norma, no_mean=no_mean)
    Scale = np.linspace(2*np.pi / 2.6, 45*size//128 * 2 * np.pi /2.6 , 45*size//128)
    
    fig, ax = plt.subplots(figsize=(16,13), facecolor="white")
    colors = ['red','navy','green','royalblue','darkorange','lightskyblue','darkgray','limegreen','peru']

    for j in range(N_var):
        ax.plot(np.log10(Scale), np.log10(spectrum_arome[j,:]), linestyle='--', 
                linewidth = 2.5, color=colors[j], label = f'AROME - {var_names[j]}')
    
    ax.set_xticks(ticks=[0.5,1.0,2.0],labels=[r'$10^{0.5}$', r'$10^{1}$', r'$10^{2}$'], size='22')   
    ax.tick_params(direction='in', length=12, width=2, pad=10)
    
    plt.yticks(fontsize='22')
    ax.set_xlabel(r'$||k||$ $(km^{-1})$', fontsize = '25',fontweight ='bold')
    
    ax.legend(fontsize='20',frameon = False)
    ax.set_ylabel(r'$\log_{10}PSD(k)$', fontsize = 25)
    with_mean = "normal" if not no_mean else "without mean"
    ax.set_title(f"Arome spectrum ({with_mean}) for different variables on the {size}x{size} domain", fontsize=25)
    fig.tight_layout()
    save_name = filepath+f"Arome_{'_'.join(var for var in var_names)}_{size}_comparison"
    save_name += "_no_mean.png" if no_mean else ".png"
    fig.savefig(save_name, dpi=400)
        


def spectral_plot(program, output_dir, var_names=['u','v','t2m'], lat=512, bs=16, chm=2, norma='classic', no_mean=False,
        lr=0.002, ckpt=147000, use_noise=True, size=128, N_samp=65536, rgb_level=3, res=[128], mean_pert=False, zoom=False):
    
    assert not (no_mean and mean_pert), "You can't have no_mean and mean_pert activated at the same time"
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    """
    plot 'fake' spectra against real spectrum data
    save figure in output_dir
    compute PSD error for each variable specified in var_names (explicit names are important here)
    print this error
    """
    var_names = sorted(var_names, key=lambda var:var_dict[var])
    N_vars = len(var_names)
    
    Scale = np.linspace(2*np.pi / 2.6, 45*size//128 * 2 * np.pi /2.6 ,45*size//128) # ici faut faire gaffe ça marche avec 128,256 mais peut être pas au delà
    
    spectrum_arome = load_arome_spectrum(var_names, size=size, norma=norma, no_mean=no_mean)
    spectrum = load_gan_spectrum(program=program, vars=var_names, lat=lat, bs=bs, chm=chm, lr=lr, ckpt=ckpt, use_noise=use_noise, 
            size=size, N_samp=N_samp, rgb_level=rgb_level, res=res, norma=norma, no_mean=no_mean, mean_pert=mean_pert)
    spec_col = ['red','navy','green','royalblue','darkorange','crimson','darkgray','limegreen','peru']
    
    size_str = f"{size}x{size}"
      
    for j in range(N_vars):
        name = f"rgb_lvl_{rgb_level}_{size_str}_{lat}_{bs}_{chm}_{lr}_{'_'.join(var for var in var_names)}_{use_noise}"+\
            f"_res_{'_'.join(str(r) for r in res)}_norma_{norma}_{var_names[j]}_{ckpt}" + ("_no_mean" if no_mean else "")
        name += "_mean_pert" if mean_pert else ""

        fig, ax1 = plt.subplots(figsize = (12,11), facecolor='white')
        
        for e, r in enumerate(res):        
            ax1.plot(np.log10(Scale), np.log10(spectrum[e,j,:]), color=spec_col[e], linestyle='--',
                     linewidth = 2.5, label  = f"StyleGAN({multiline_label(','.join(var for var in var_names))})\n"+\
                        f"domain ({r},{r})")
            if zoom and var_names[j]!='z500':
                ax2 = zoomed_inset_axes(ax1, zoom=1.5, loc="upper right")
                ax2.plot(np.log10(Scale), np.log10(spectrum[e,j,:]), color=spec_col[e], linestyle='--',
                        linewidth = 2.5)

        ax1.plot(np.log10(Scale), np.log10(spectrum_arome[j,:]), 'kd', 
                linewidth = 2.5, label = 'AROME')
        if zoom and var_names[j]!='z500':
            ax2.plot(np.log10(Scale), np.log10(spectrum_arome[j,:]), 'kd', 
                linewidth = 2.5)
        ax1.set_title("Spectrum comparison: " + var_names[j] + (" (no mean) " if no_mean else "") +\
                      (" (mean/pert)" if mean_pert else ""), fontsize = 25)
     
        ax1.set_xticks(ticks=[0.5,1.0,2.0],labels=[r'$10^{0.5}$', r'$10^{1}$', r'$10^{2}$'], size='22')
        if zoom and var_names[j]!='z500':
            ax2.set_xticks(ticks=[2.0],labels=[r'$10^{2}$'], size='22')
        
        ax1.tick_params(direction='in', length=12, width=2, pad=10)
        if zoom and var_names[j]!='z500':
            ax2.tick_params(direction='in', length=12, width=2, pad=10)
        
        
        plt.yticks(fontsize='22')
        # voir le pad des ticks sur x
        ax1.set_xlabel(r'$||k||$ $(km^{-1})$', fontsize = '25',fontweight ='bold')
        
        ax1.legend(fontsize='20',frameon = False, loc = "lower left" if zoom else "upper right")
        ax1.set_ylabel(r'$\log_{10}PSD(k)$', fontsize = 25)

        if zoom and var_names[j]!='z500':
            x1,y1,x2,y2 = 1.6, np.min(np.log10(spectrum_arome[j,:]))-0.1, np.log10(Scale)[-1]+0.02, -6.5
            ax2.set_xlim(x1, x2)
            ax2.set_ylim(y1, y2)
            ax2.set_yticks([np.ceil(y1),np.floor(y2)])
            mark_inset(ax1, ax2, loc1=4, loc2=3, fc="none", ec="0.5")
    
        fig.tight_layout()
        fig.savefig(output_dir+f"Spectral_PSD_{name}.png", pad_inches=1, dpi=400)
        
    return 0


# TODO: add possibility to use spec with mean/pert
def compare_spectrums_gan(program, output_dir, var_names=[['u','v','t2m']], lat=[512], bs=[16], chm=[2], norma='classic',
        lr=[0.002], ckpt=[147000], use_noise=[True], size=128, N_samp=65536, rgb_level=[3], res=[128], mean_pert=[False], zoom=False):
    
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

    dic = {'m/pert': mean_pert, 'noise': use_noise,'lat': lat, 'bs': bs, 'chm': chm, 'lr': lr, 'ckpt': ckpt}
    nb_spec = len(dic[max(dic, key=lambda x:len(dic[x]))])
    modes = []
    modes_list = []
    for k in dic.keys():
        if len(dic[k])>=nb_spec and k!='ckpt':
            modes.append(k)
            modes_list.append(dic[k])
        dic[k] = [dic[k][0] for i in range(nb_spec)] if len(dic[k])==1 else dic[k]
    modes_print = [', '.join([f"{modes[i]}: {modes_list[i][j]}" for i in range(len(modes))]) for j in range(nb_spec)]
    
    Scale = np.linspace(2*np.pi / 2.6, 45*size//128 * 2 * np.pi /2.6 ,45*size//128) # ici faut faire gaffe ça marche
                                                                                    # avec 128,256 mais peut être pas au delà
    var_names = [var_names[0] for i in range(nb_spec)] if len(var_names)==1 else var_names
    idx = var_names.index(min(var_names, key=len))
    var_names = [sorted(var_names[i], key=lambda var:var_dict[var]) for i in range(len(var_names))]
    spectrum_arome = load_arome_spectrum(var_names[idx], size=size, norma=norma)
    spectrums_gan = []

    for i in range(nb_spec):
        spectrums_gan.append(load_gan_spectrum(program=program, vars=var_names[i], lat=dic['lat'][i], bs=dic['bs'][i], 
                chm=dic['chm'][i], lr=dic['lr'][i], ckpt=dic['ckpt'][i], use_noise=dic['noise'][i], size=size, 
                N_samp=N_samp, rgb_level=rgb_level, res=res, norma=norma, mean_pert=dic["m/pert"][i]))
    
    spec_col = ['red','navy','green','royalblue','darkorange','crimson','darkgray','peru','limegreen']
    
    size_str = f"{size}x{size}"
      
    for i, var in enumerate(var_names[idx]):
        ids = [var_names[j].index(var) for j in range(nb_spec)]
        l, b, c, Lr, n, ck, mp = '_'.join(str(la) for la in lat), '_'.join(str(bat) for bat in bs), '_'.join(str(ch) for ch in chm), \
                         '_'.join(str(l) for l in lr), '_'.join(str(n) for n in use_noise), '_'.join(str(ckp) for ckp in ckpt), \
                         '_'.join(str(m_p) for m_p in mean_pert)
        
        name = f"spec_comparison_{rgb_level}_{size_str}_l_{l}_b_{b}_ch_{c}_lr_{Lr}_{'_'.join(var for var in var_names[idx])}"+\
            f"_n_{n}_res_{'_'.join(str(r) for r in res)}_norma_{norma}_{var}_{ck}_mp_{mp}"

        fig, ax1 = plt.subplots(figsize = (12,11), facecolor='white')
        
        for e, r in enumerate(res):
            for j in range(nb_spec):
                ax1.plot(np.log10(Scale), np.log10(spectrums_gan[j][e,ids[j],:]), color=spec_col[j], linestyle='--',
                     linewidth = 2.5, label  = f"StyleGAN({multiline_label(','.join(var for var in var_names[0]))})\n"+\
                        f"domain ({size},{size}), {modes_print[j]}")
                if zoom and var!='z500':
                    ax2 = zoomed_inset_axes(ax1, zoom=1.5, loc="upper right")
                    ax2.plot(np.log10(Scale), np.log10(spectrums_gan[j][e,ids[j],:]), color=spec_col[j], linestyle='--',
                            linewidth = 2.5)
            
        ax1.plot(np.log10(Scale), np.log10(spectrum_arome[i,:]), 'kd', 
                linewidth = 2.5, label = 'AROME')
        if zoom and var!='z500':
            ax2.plot(np.log10(Scale), np.log10(spectrum_arome[i,:]), 'kd', 
                linewidth = 2.5)

        ax1.set_title("Spectrum comparison: " + var + f"\ncheckpoints: {', '.join([str(ck) for ck in ckpt])}", fontsize = 25)
     
        ax1.set_xticks(ticks=[0.5,1.0,2.0],labels=[r'$10^{0.5}$', r'$10^{1}$', r'$10^{2}$'], size='22')
        if zoom and var!='z500':
            ax2.set_xticks(ticks=[2.0],labels=[r'$10^{2}$'], size='22')
        
        ax1.tick_params(direction='in', length=12, width=2, pad=10)
        if zoom and var!='z500':
            ax2.tick_params(direction='in', length=12, width=2, pad=10)
        
        plt.yticks(fontsize='22')
        ax1.set_xlabel(r'$||k||$ $(km^{-1})$', fontsize = '25',fontweight ='bold')
        
        ax1.legend(fontsize='20',frameon = False, loc = "lower left" if zoom else "upper right")
        ax1.set_ylabel(r'$\log_{10}PSD(k)$', fontsize = 25)

        if zoom and var!='z500':
            x1,y1,x2,y2 = 1.6, np.min(np.log10(spectrum_arome[i,:]))-0.1, np.log10(Scale)[-1]+0.02, -6.5
            ax2.set_xlim(x1, x2)
            ax2.set_ylim(y1, y2)
            ax2.set_yticks([np.ceil(y1),np.floor(y2)])
            mark_inset(ax1, ax2, loc1=4, loc2=3, fc="none", ec="0.5")
    
        fig.tight_layout()
        fig.savefig(output_dir+f"Spectral_PSD_{name}.png", pad_inches=1, dpi=400)



# to be changed 
def plot_multi_vs_mono(spectrums, output_dir, varss=[['u','v','t2m'],['u','v'],['t2m']], res=[8,32,128]):
    assert len(spectrums)==len(varss), "You should have one spectrum per set of variable"
    
    all_var = []
    for vars in varss:
        all_var += vars
    all_var = list(np.unique(all_var))
    all_var = sorted(all_var, key=lambda var:var_dict[var])
    N_vars = len(all_var)
    
    Scale = np.linspace(2*np.pi / 2.6, 45 * 2 * np.pi /2.6 ,45)
    
   
    spec_col = ['red','navy','green','royalblue','darkorange','crimson','darkgray','limegreen','peru']
    spec_markers = ['o', 'v', 'd']
    marker_edge_colors = ['limegreen', 'peru', 'darkgray', 'crimson']
    linestyles = [(0, (2, 2)), (0, (4, 4)), (0, (6,6))]
    markeverys = [4,5,6]
    for j in range(N_vars):
       
        fig, ax = plt.subplots(figsize = (9,7), facecolor='white')
        
        #print(np.log10(spectrum[j,:]))
        #print(np.log10(Scale))
        for e, r in enumerate(res):
            for s, spec in enumerate(spectrums):
                if all_var[j] in varss[s]:
                    j_idx = [varss[s].index(i) for i in varss[s] if all_var[j] in i]
                    ax.plot(np.log10(Scale), np.log10(spec[e,j_idx[0],:]), color=spec_col[e], marker=spec_markers[s],
                    linewidth = 2., linestyle=linestyles[s],markevery=markeverys[s],#markeredgecolor=marker_edge_colors[s], 
                    label  = f"StyleGAN({','.join(var for var in varss[s])}) at res ({r},{r})")
        ax.set_title(all_var[j], fontsize = 25)        
        #plt.grid()
     
        ax.set_xticks(ticks=[0.5,1.0,2.0],labels=[r'$10^{0.5}$', r'$10^{1}$', r'$10^{2}$'], size='22')
        
        ax.tick_params(direction='in', length=12, width=2, pad=10)
        
        
        plt.yticks(fontsize='22')
        ax.set_xlabel(r'$||k||$ $(km^{-1})$', fontsize = '25',fontweight ='bold')
        
        ax.legend(fontsize='15',frameon = False)
        ax.set_ylabel(r'$\log_{10}PSD(k)$', fontsize = 25)
    
        fig.tight_layout()
        #plt.show()
        fig.savefig(output_dir+f"Spectral_PSD_{all_var[j]}_multi_mono_{'|'.join('_'.join(var for var in vars) for vars in varss)}"\
                    +f"{'_'.join(str(r) for r in res)}.png", pad_inches=1, dpi=400)
        #fig.close()
        
    return 0





# not used
def spectral_plot_simple(spectrum, spectrum_2, output_dir, step=None, var_names=['u','v','t2m']):
    """
    plot 'fake' spectra against real spectrum data
    save figure in output_dir
    compute PSD error for each variable specified in var_names (explicit names are important here)
    print this error
    """
    
    N_vars = len(var_names)
    
    Scale = np.linspace(2*np.pi / 2.6, 90 * 2 * np.pi /2.6 ,45)
        
    for j in range(N_vars):
       
        fig,axs = plt.subplots(figsize = (9,7))
        
        print(np.log10(spectrum[j,:]))
        print(np.log10(Scale))
                
        plt.plot(np.log10(Scale), np.log10(spectrum[j,:]), 'r--', linewidth = 2.5, label  = 'StyleGAN')
        plt.plot(np.log10(Scale), np.log10(spectrum_2[j,:]), 'kd', linewidth = 4.5, label = 'AROME')
        plt.title(var_names[j], fontsize = 25)        
        #plt.grid()
     
        plt.xticks([0.5,1.0,2.0],[r'$10^{0.5}$', r'$10^{1}$', r'$10^{2}$'], fontsize ='22')
        
        axs.tick_params(direction='in', length=12, width=2)
        
        
        plt.yticks(fontsize='22')
        plt.xlabel(r'$||k||$ $(km^{-1})$', fontsize = '25',fontweight ='bold')
        if j==0 :
            plt.legend(fontsize='25',frameon = False)
            plt.ylabel(r'$\log_{10}PSD(k)$', fontsize = 25)
    
        fig.tight_layout()
        plt.show()
        #plt.savefig(output_dir+'Spectral_PSD_{}.png'.format(var_names[j]))
        plt.close()
        
    return 0





# not used
def plot_abs_err_multi_vs_mono(spectrums, output_dir, varss=[['u','v','t2m'],['u','v'],['t2m']], res=[8,32,128]):
    assert len(spectrums)==len(varss), "You should have one spectrum per set of variable"
    
    all_var = []
    for vars in varss:
        all_var += vars
    all_var = list(np.unique(all_var))
    N_vars = len(all_var)
    
    Scale = np.linspace(2*np.pi / 2.6, 45 * 2 * np.pi /2.6 ,45)
    
   
    spec_col = ['red','navy','green','royalblue','darkorange','crimson','darkgray','limegreen','peru']
    spec_markers = ['o', 'v', 'd']
    marker_edge_colors = ['limegreen', 'peru', 'darkgray', 'crimson']
    linestyles = [(0, (2, 2)), (0, (4, 4)), (0, (6,6))]
    markeverys = [4,5,6]
    for j in range(N_vars):
       
        fig, ax = plt.subplots(figsize = (9,7), facecolor='white')
        
        #print(np.log10(spectrum[j,:]))
        #print(np.log10(Scale))
        for e, r in enumerate(res):
            x = 0.0
            lab = "|"
            counter = 0
            for s, spec in enumerate(spectrums):
                if all_var[j] in varss[s]:
                    counter += 1
                    j_idx = [varss[s].index(i) for i in varss[s] if all_var[j] in i]
                    y = spec[e,j_idx[0],:]
                    x = np.abs(x-y)
                    lab += f"StyleGAN({','.join(var for var in varss[s])})"
                    if counter==1:
                        lab += "-"
            lab += f"| at res ({r},{r})"
            ax.plot(np.log10(Scale), x, color=spec_col[e], marker=spec_markers[s],
                    linewidth = 2., linestyle=linestyles[s],markevery=markeverys[s],#markeredgecolor=marker_edge_colors[s], 
                    label  = lab)
            ax.set_yscale('log')
        ax.set_title(all_var[j], fontsize = 25)        
        #plt.grid()
     
        ax.set_xticks(ticks=[0.5,1.0,2.0],labels=[r'$10^{0.5}$', r'$10^{1}$', r'$10^{2}$'], size='22')
        
        ax.tick_params(direction='in', length=12, width=2, pad=10)
        
        
        plt.yticks(fontsize='22')
        ax.set_xlabel(r'$||k||$ $(km^{-1})$', fontsize = '25',fontweight ='bold')
        
        ax.legend(fontsize='15',frameon = False)
        ax.set_ylabel(r'$PSD(k)$', fontsize = 25)
    
        fig.tight_layout()
        #plt.show()
        fig.savefig(output_dir+f"Spectral_PSD_{all_var[j]}_abs_err_multi_mono_{'|'.join('_'.join(var for var in vars) for vars in varss)}"\
                    +f"{'_'.join(str(r) for r in res)}.png", pad_inches=1, dpi=400)
        #fig.close()
        
    return 0

if __name__=='__main__':
    var_names = ['u','v','t2m','z500','t850','tpw850']
    size = 128
    no_mean = True
    plot_arome_spectrum(var_names=var_names, size=size, norma='classic', no_mean=no_mean)