#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:23:42 2022

@author: brochetc


Many plotting functions for (hopefully) artistic rendering individual metrics plots

"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import multivariate as mlt


"""
######################  How to use this file ##################################

This file is intended to plot results from metrics calculations,
contained in dictionaries

These dictionaries are pickle objects with extension .p
They are typically opened with pickle.load(open('myFile.p', 'rb'))

The dict is then organized like :
    {'metric' : np.array(Metric)}
'Metric' is the name of the metric (cf metrics4arome package)
The numpy array may be ANY shape, then it should be referred to the metric doc
in order to exploit it.

HOWEVER, a common organization of the numpy is the following :
    
    np.array(Metric)  of shape S x C x d1 x d2..
    
    with S the number of Steps (ie  : iteration of the training algorithm)
    on which the metric is computed. Typical S  = approx. 40 
    (ie each 1500 in a 60k steps training)
    
    and C the number of variables 
    (so one often -but not always- has one metric per variable)

So each function basically takes a dictionary as argument, finds the metric
corresponding to the function, extracts data at the given 'step' = index of first dimension
and then plots metrics in a beautiful way. step is None by default, meaning default behaivor assume
Step dimension doesn't exist

"""


var_names = ['u', 'v', 't2m']

def wasserstein_maps(dictionary, output_dir, step=None, var_names = ['u','v','t2m']):
    """
    plot wasserstein maps contained in dictionary
    save figure in output_dir
    
    var_names is a list containing the number of variables one wants to plot
    (name for the variables is not important per se but explicit is better than implicit)
    """
    
    N_vars = len(var_names)
    
    if step is not None :
        PW = dictionary[step]['pw_W1']
        PW = np.array(PW)
        print(len(PW),PW[0].shape)
    else :
        PW = dictionary['pw_W1']
        
    fig = plt.figure(figsize = (6,3*N_vars))

    grid = ImageGrid(fig, 111,
                    nrows_ncols=(1, N_vars),
                    axes_pad=(0.35, 0.35),
                    label_mode="L",
                    share_all=True,
                    cbar_location="bottom",
                    cbar_mode="edge",
                    cbar_size="10%",
                    cbar_pad="20%",
                    )
    
    axes = []
    
    for i in range(N_vars):
    
        ax = grid[i]
        axes.append(ax)
        im = ax.imshow(1e3*PW[:,i,:,:].mean(axis=0), origin ='lower', cmap = 'coolwarm')
        cb = ax.cax.colorbar(im)
    
    
    fig.tight_layout()
    plt.savefig(output_dir+f'W1_map_comparison_{step}.png')
    plt.close()
    return 0

def spectral_plot(dictionary, real_spectrum, output_dir, step=None, var_names=['u','v','t2m']):
    """
    plot 'fake' spectra against real spectrum data
    save figure in output_dir
    compute PSD error for each variable specified in var_names (explicit names are important here)
    print this error
    
    
    """
    
    N_vars = len(var_names)
    
    if step is not None :
        fake_spectrum = dictionary['spectral_compute'][step]
        
    else :
        fake_spectrum = dictionary['spectral_compute']
    
    Scale = np.linspace(2*np.pi / 2.6, 90 * 2 * np.pi /2.6 ,90)
    psd = np.zeros((N_vars,))
      
    for j in range(N_vars):
    
        fig = plt.figure(figsize=(18,15))
        
        
        """ lab = 'Mean GAN Spectrum'
        
        plt.plot(Scale, np.log10(fake_spectrum[j,:]), 'r-', linewidth = 2.5, label=lab)
                
        plt.grid()

        plt.xticks([0.5,1.0,2.0],[r'$10^{2}$', r'$10$', r'$10^{0.5}$'], fontsize ='28')
        plt.yticks(fontsize='28')
        plt.xlabel(r'Spatial Scale ($km$)', fontsize = '32',fontweight ='bold')
    
        plt.plot(Scale, np.log10(real_spectrum[j,:,0]), 'k-', linewidth=2.5, label='Mean AROME Spectrum')
        plt.plot(Scale, np.log10(real_spectrum[j,:,1]), 'k--', linewidth=2.5,label='Q10-90 AROME-EPS')
        plt.plot(Scale, np.log10(real_spectrum[j,:,2]), 'k--', linewidth=2.5, label='Q10-90 AROME-EPS')
    
    
        fig.tight_layout()
        plt.savefig(output_dir+'Spectral_PSD_{}.png'.format(var_names[j]))
        plt.close()
        
        spec = fake_spectrum[j]
        spec0_real = real_spectrum[j,:,0]
        psd_rmse = 10*np.sqrt(np.mean(((np.log10(spec)-np.log10(spec0_real))**2)))
        print('PSD error {} in dB'.format(var_names[j]), psd_rmse)"""

        plt.plot(np.log10(Scale), np.log10(fake_spectrum[j,:]), 'r--', linewidth = 2.5, label  = 'StyleGAN')
        plt.plot(np.log10(Scale), np.log10(real_spectrum[j,:]), 'kd', linewidth = 4.5, label = 'AROME')
        plt.title(var_names[j], fontsize = 25)        
        #plt.grid()
        
        plt.xticks([0.5,1.0,2.0],[r'$10^{0.5}$', r'$10^{1}$', r'$10^{2}$'], fontsize ='22')
        
        #axs.tick_params(direction='in', length=12, width=2)
        
        
        plt.yticks(fontsize='22')
        plt.xlabel(r'$||k||$ $(km^{-1})$', fontsize = '25',fontweight ='bold')
        if j==0 :
            plt.legend(fontsize='25',frameon = False)
            plt.ylabel(r'$\log_{10}PSD(k)$', fontsize = 25)

        fig.tight_layout()
        #plt.show()
        plt.savefig(output_dir+'Spectral_PSD_{}_{}.png'.format(var_names[j], step))
        plt.close()


        psd = 10 * np.sqrt(np.mean((np.log10(fake_spectrum) - np.log10(real_spectrum))**2, axis=1))
    return psd


def length_scales_maps(dictionary, real_corr,  output_dir, step = None, variable_names=['u','v','t2m']):
    """
    plot correlation length scales maps using dictionary 
    against real lengths contained in real_corr
    
    save figure in output_dir
    
    compute statistics of Mean Absolute error between real and fake on the maps
    and print those statistics
    
    """
    if step is not None :
        fake_corr_length = dictionary['ls_metric'][step]
        
    else :
        fake_corr_length = dictionary['ls_metric']
    
    print(fake_corr_length.shape)
    
    
    N_vars = len(var_names)
    
    fig = plt.figure(figsize = (6,3*N_vars))

    grid = ImageGrid(fig, 111,
                    nrows_ncols=(2, N_vars),
                    axes_pad=(0.35, 0.35),
                    label_mode="L",
                    share_all=True,
                    cbar_location="bottom",
                    cbar_mode="edge",
                    cbar_size="10%",
                    cbar_pad="20%",
                    )
    
    for i in range(N_vars) : 
    
        ax4 = grid[N_vars+i]
        im4 = ax4.imshow(fake_corr_length[i,:,:], origin ='lower')
        
        ax = grid[i]
        ax.imshow(real_corr[i,:,:], origin ='lower', vmin = fake_corr_length[i].min(), vmax = fake_corr_length[i].max())
        
        cb4 = ax4.cax.colorbar(im4)
        
        
    fig.tight_layout()
    plt.savefig(output_dir+f'Length_scales_map_comparison_{step}.png')
    plt.close()
    
    ######### computing statistics
    
    diff_corr = real_corr - fake_corr_length
    
    mae_corr = np.abs(diff_corr).mean(axis=(-2,-1))
    std_ae_corr = np.abs(diff_corr).std()
    max_ae_corr = np.abs(diff_corr).max()
    
    print('MAE Lcorr', mae_corr.mean())
    print('STD AE corr', std_ae_corr)
    print('Max ae corr', max_ae_corr)
    
    return mae_corr


def multivariate_correlation(dictionary, output_dir, step = None, var_names=['u','v','t2m']):
    """
    plot multivariate data correlation distributions contained in dictionary (reference to real data also in dictionary)
    
    save figures in output_dir
    
    """
    
    if step is None or dictionary[step]['multivar'][0].shape[0]==1:
        RES = dictionary['multivar'].squeeze()
    else :
        print(dictionary[step].keys())
        RES = dictionary[step]['multivar']
    
    for i in range(1):
        print(i)
        data_r,data_f = RES[i][0], RES[i][1]
        print(data_r.shape, data_f.shape)
        
        
        levels = mlt.define_levels(data_r,5)  #defining color and contour levels
        
        ncouples2 = data_f.shape[0]*(data_f.shape[0]-1)
        
        bins = np.linspace(tuple([-1 for i in range(ncouples2)]), tuple([1 for i in range(ncouples2)]),101, axis=1)
        
        var_r = (np.log(data_r), bins)
        var_f = (np.log(data_f), bins)
        
        mlt.plot2D_histo(var_f, var_r, levels, output_dir = output_dir, add_name=f'{step}')
