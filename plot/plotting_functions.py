#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:03:46 2022

@author: brochetc

Plotting Functions for 2D experiments

"""


import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import empty
import torch
from numpy import log10, histogram
import pandas as pd
import numpy as np
import pickle
import os
from glob import glob
import random
import re
import math


################ reference dictionary to know what variables to sample where
################ do not modify unless you know what you are doing 
var_dict={'rr' : 0, 'u' : 1, 'v' : 2, 't2m' :3 , 'orog' : 4, 'z500': 5, 't850': 6, 'tpw850': 7}
var_dict_unit={'rr': "mm", 'u': "m/s", 'v': "m/s", 't2m': "K" , 'orog': "m", 'z500': "m", 't850': "K", 'tpw850': "K"}



def flatten(arg):
    # flattent list of any depth into a list of depth 1
    if not isinstance(arg, list): # if not list
        return [arg]
    return [x for sub in arg for x in flatten(sub)] # recurse and collect


def multiline_label(label):
    # to write long labels on several lines
    if type(label)!=list:
        return label
    new_l = ''
    s = 0
    s_max = len(''.join(str(l) for l in label))
    cut = s_max // 2 if s_max > 11 else s_max
    for lab in label:
        s += len(lab)
        if s > cut:
            if lab != label[-1]:
                new_l += '\n'+lab+', '
            else:
                new_l += '\n'+lab
            s = 0
        else:
            if lab != label[-1]:
                new_l += lab + ', '
            else:
                new_l += lab
    return new_l

def online_distrib_plot(epoch,n_samples, train_samples, modelG, device):
    
    z=empty((n_samples,modelG.nz)).normal_().to(device)
    modelG.train=False
    out_fake=modelG(z).cpu().detach().numpy()
    modelG.train=True
    data_list=[train_samples.cpu().detach(), out_fake]
    legend_list=["Data", "Generated"]
    var_names=["rr" , "u", "v", "t2m"]
    title="Model Performance after epoch "+str(epoch+1)
    option="climato"
    plot_distrib_simple(data_list, legend_list, var_names, title, option)
    
    
def plot_distrib_simple(data_list,legend_list, var_names,title, option):
    
    """
    plot the distribution of data -- one distribution for each value of the last axis
    """
    fig=plt.figure(figsize=(6,8))
    st=fig.suptitle(title+" "+option, fontsize="x-large")
    data=data_list[0]
    N_var=data.shape[-1]
    columns=1
    for i in range(N_var):
        ax=plt.subplot(N_var, columns, i+1)
        for j,data in enumerate(data_list):
            if var_names[i].find('rr')!=-1:
                o1,o2=histogram(data[:,i], bins=200)
                o2_=o2[:-1]
                o1log=log10(o1)
                ax.plot(o2_, o1log, 'o', label=legend_list[j])
            else :
                ax.hist(data[:,i], bins=200, density=True, label=legend_list[j])
        ax.set_ylabel(var_names[i])
        ax.legend()
    fig.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.9)
    plt.savefig(title+"_"+option+".png")
    plt.show()
    plt.close()

    return 0

    
def plot_GAN_metrics(metrics_list):
    fig=plt.figure(figsize=(6,8))
    st=fig.suptitle("Metrics", fontsize="x-large")
    rows=len(metrics_list)
    columns=1
    for i,metric in enumerate(metrics_list) :
        ax=plt.subplot(rows, columns, i+1)
        ax.plot(range(1,len(metric.data)+1,1),metric.data)
        
        ylabel=metric.name
        ax.set_ylabel(ylabel)
    plt.xlabel("Number of epochs")
    fig.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.9)
    plt.savefig("GAN_metrics_graph.png")
    plt.close()
    
def online_sample_plot(batch, n_samples, Step, var_names, path, mean_pert=False):
    
    batch_to_print = batch[:n_samples]
    IMG_SIZE = batch.shape[2]
    for i, var in enumerate(var_names):
        if var=='t2m':
            varname='2m temperature'
            cmap='coolwarm'
            limits = (-0.5,0.5)

        elif var=='rr':
            varname='Rain rate'
            cmap='Blues'
            limits = (-0.1,0.1)
            
        elif var=='orog':
            varname='Orography'
            cmap='terrain'
            limits = (-0.95,0.95)
        
        elif var=='z500':
            varname='500 hPa geopotential'
            cmap='Blues'
            limits = (batch_to_print[:,i,:,:].min(), batch_to_print[:,i,:,:].max())
        
        elif var=='t850':
            varname='850 hPa temperature'
            cmap='coolwarm'
            limits = (-0.5,0.5)
            
        elif var=='tpw850':
            varname='tpw850'
            cmap='plasma'
            limits = (-0.5,0.5)
                    
        else :
            varname='Wind '+var
            cmap='viridis'
            limits = (-0.5,0.5)
            
        fig = plt.figure(figsize=(20,20))
        rows = 4
        columns = 4
        st = fig.suptitle(varname + (" pert" if mean_pert else ""), fontsize='30')
        st.set_y(0.96)
        
        for j in range(batch_to_print.shape[0]) :
            b = batch_to_print[j][i].view(IMG_SIZE, IMG_SIZE)
            ax = fig.add_subplot(rows, columns, j+1)
            im = ax.imshow(b.cpu().detach().numpy()[::-1,:], cmap=cmap, 
                         vmin=limits[0], vmax=limits[1])
            
        fig.subplots_adjust(bottom=0.05,top=0.9, left=0.05, right=0.9)
        cbax=fig.add_axes([0.92,0.05,0.02,0.85])
        cb=fig.colorbar(im, cax=cbax)
        cb.ax.tick_params(labelsize=20)
        plt.savefig(path+"/Samples_at_Step_"+str(Step)+'_'+var+("_pert" if mean_pert else "")+".png")
        plt.close()

        if mean_pert:
            fig = plt.figure(figsize=(20,20))
            rows = 4
            columns = 4
            st = fig.suptitle(varname + " mean", fontsize='30')
            st.set_y(0.96)
            
            for j in range(batch_to_print.shape[0]) :
                b = batch_to_print[j][i+len(var_names)].view(IMG_SIZE, IMG_SIZE)
                ax = fig.add_subplot(rows, columns, j+1)
                im = ax.imshow(b.cpu().detach().numpy()[::-1,:], cmap=cmap, 
                            vmin=limits[0], vmax=limits[1])
                
            fig.subplots_adjust(bottom=0.05,top=0.9, left=0.05, right=0.9)
            cbax=fig.add_axes([0.92,0.05,0.02,0.85])
            cb=fig.colorbar(im, cax=cbax)
            cb.ax.tick_params(labelsize=20)
            plt.savefig(path+"/Samples_at_Step_"+str(Step)+'_'+var+("_mean" if mean_pert else "")+".png")
            plt.close()

    return 0

def plot_metrics_from_csv(log_path,filename, metrics_list=[]):
    """
    file structure should be 'Step,metric1,metric2,etc...'
    """
    df=pd.read_csv(log_path+filename)
    if len(metrics_list)==0:
        metrics_list=df.columns[1:]
    N_metrics=len(df.columns[1:])

    figure=plt.figure(figsize=(36,(N_metrics//3+1)*8), facecolor='white')
    st=figure.suptitle("Metrics", fontsize="xx-large")
    for i,metric in enumerate(metrics_list):
        ax=plt.subplot(N_metrics//3+1,3,i+1)
        ax.plot(df['Step'], df[metric])
        ax.set_ylabel(metric, fontsize="x-large")
    plt.xlabel('Iteration step')
    figure.tight_layout()
    st.set_y(0.95)
    figure.subplots_adjust(top=0.9)
    plt.savefig(log_path+"GAN_metrics_graph.png", bbox_inches='tight', dpi=200)
    plt.close()
    
def plot_metrics_on_same_plot(log_path,filename, metrics_list=[], targets=[]):
    """
    file structure should be 'Step,metric1,metric2,etc...'
    """
    df=pd.read_csv(log_path+filename)
    if len(metrics_list)==0:
        metrics_list=df.columns[1:]
    
    colors=['b','r','k', 'g', 'orange']

    figure=plt.figure(figsize=(10,10))
    st=figure.suptitle("Metrics", fontsize="x-large")
    for i,metric in enumerate(metrics_list):
        if metric=='criterion':
            label='W1_crop'
        else: 
            label=metric
        plt.plot(df['Step'], df[metric], label=label, color=colors[i])
    plt.hlines(targets,xmin=0,xmax=49000,colors=colors[:len(metrics_list)], linestyles='--')
    plt.grid()
    plt.legend()
    #plt.set_ylabel('Distance')
    plt.xlabel('Iteration step')
    plt.yscale('log')
    figure.tight_layout()
    st.set_y(0.95)
    figure.subplots_adjust(top=0.9)
    plt.savefig(log_path+"GAN_metrics_graph.png")
    plt.close()


def compute_final(set_path, steps, f_name, add_name='_16384_16384', mode='lat', lat_dim=64, chm=2, bs=16, 
                    lr=0.002, vars=['u','v','t2m'], use_noise=True, dom_size=128, no_mean=False, mean_pert=False):

    print("Computing final")
    whole_d = f"_dom_{dom_size}"
    expe_path = f"stylegan2_stylegan_dom_{dom_size}_lat-dim_{lat_dim}_bs_{bs}_{lr}_{lr}_ch-mul_{chm}"\
                    + f"_vars_{'_'.join(str(var) for var in vars)}_noise_{use_noise}" 
    expe_path += ("_no_mean" if no_mean else "") + ("_mean_pert" if mean_pert else "")
    expe_path += "/Instance_1/log/"
    set_path_init = set_path
    set_path_final = set_path + f'Comparison_{mode}/Dom_{dom_size}/'
    if mode=='lat':
        filename = set_path_final + f_name + '_'.join([str(s) for s in steps]) +\
             f"{whole_d}_lat_{lat_dim}_bs_{bs}_chm_{chm}_var_{'_'.join(str(var) for var in vars)}_noise_{use_noise}" +\
                 add_name + '_final.p'
    elif mode=='bs':
        filename = set_path_final + f_name + '_'.join([str(s) for s in steps]) +\
             f"{whole_d}_lat_{lat_dim}_bs_{bs}_chm_{chm}_var_{'_'.join(str(var) for var in vars)}_noise_{use_noise}" +\
                 add_name + '_final.p'
    elif mode=='chm':
        filename = set_path_final + f_name + '_'.join([str(s) for s in steps]) +\
             f"{whole_d}_lat_{lat_dim}_bs_{bs}_chm_{chm}_var_{'_'.join(str(var) for var in vars)}_noise_{use_noise}" +\
                 add_name + '_final.p'
    elif mode=='var':
        filename = set_path_final + f_name + '_'.join([str(s) for s in steps]) +\
             f"{whole_d}_lat_{lat_dim}_bs_{bs}_chm_{chm}_var_{'_'.join(str(var) for var in vars)}_noise_{use_noise}" +\
               ("_mean_pert" if mean_pert else "") + ("_no_mean" if no_mean else "") + add_name + '_final.p'
    elif mode=='noise':
        filename = set_path_final + f_name + '_'.join([str(s) for s in steps]) +\
             f"{whole_d}_lat_{lat_dim}_bs_{bs}_chm_{chm}_var_{'_'.join(str(var) for var in vars)}_noise_{use_noise}" +\
                 add_name + '_final.p'
    else:
        print("something is wrong with your call (if you used domain as mode you need to first compute the\
               .p with another mode and then move it to the correct folder)")
        return 0

    final_dic = {}
    if not os.path.isfile(filename):
        for s in steps:
            dict_path = set_path_init + expe_path + f_name + str(s) + add_name + '.p'
            #print("dic path: ", dict_path)
            dic = np.load(dict_path, allow_pickle=True)
            #print(list(dic.keys()))
            if 'Mean' not in dic.keys() or 'Std' not in dic.keys():
                dic["Mean"] = {}
                dic["Std"] = {}
                for metric in dic[0].keys():
                    m, std = 0, 0
                    for exp in dic.keys():
                        if exp!='Mean' and exp!='Std':
                            m += dic[exp][metric]
                    m /= len(dic.keys())-2
                    for exp in dic.keys():
                        if exp!='Mean' and exp!='Std':
                            std += (dic[exp][metric]-m)**2
                    std /= len(dic.keys())-3
                    std = np.sqrt(std)
                    dic["Mean"][metric] = m
                    dic["Std"][metric] = std
                #print("   Means: ", [(metric, dic["Mean"][metric]) for metric in dic[0].keys() if metric!='multivar'], 
                #    "\n\n   Stds: ", [(metric, dic["Std"][metric]) for metric in dic[0].keys() if metric!='multivar'], '\n')
            final_dic[s] = dic
        pickle.dump(final_dic, open(filename, 'wb'))


def compare_stats_metrics(set_path, steps=[[51000,102000,147000]], add_name='_16384_16384', lat_dims=[512], chms=[2], bss=[16],
                varss=[['u','v','t2m']], use_noises=[True], dom_size=[128], no_mean=False, mean_pert=False, x_label='steps'):
    # x_label: plot with respect to nb of iterations ('steps') or images seen ('images' or anything else)
    # for now works only with mode='bs' because not relevant otherwise

    filenames = []
    f_name = "test_for_score_crawl_distance_metrics_step_"
    dic_len = {'var': len(varss),'domain': len(dom_size),'noise':len(use_noises),'lat':len(lat_dims), 'bs':len(bss), 'chm':len(chms)} 
    mode = max(dic_len, key=dic_len.get)
    
    set_path += f"Finished_experiments/"
    if len(steps)==1:
        steps = [steps[0] for i in range(dic_len[mode])]
    x_values = steps
    
    print(mode)
    if mode=='lat':
        bs = bss[0]
        chm = chms[0]
        vars = varss[0]
        noise = use_noises[0]
        label = "Lat_dim="
        filenames = [set_path + f'Comparison_{mode}/Dom_{dom_size[0]}/' + f_name + '_'.join([str(s) for s in steps[i]]) +\
                f"_dom_{dom_size[0]}_lat_{lat_dim}_bs_{bs}_chm_{chm}_var_{'_'.join(str(var) for var in vars)}_noise_{noise}" +\
                add_name + '.p' for i,lat_dim in enumerate(lat_dims)]
    elif mode=='bs':
        lat_dim = lat_dims[0]
        chm = chms[0]
        vars = varss[0]
        noise = use_noises[0]
        label= "Batch size="
        filenames = [set_path + f'Comparison_{mode}/Dom_{dom_size[0]}/' + f_name + '_'.join([str(s) for s in steps[i]]) +\
                f"_dom_{dom_size[0]}_lat_{lat_dim}_bs_{bs}_chm_{chm}_var_{'_'.join(str(var) for var in vars)}_noise_{noise}" +\
                add_name + '.p' for i,bs in enumerate(bss)]
        if not x_label.lower()=='steps':
            x_values = [np.array(steps[i])*bss[i] for i in range(len(bss))]            
    elif mode=='chm':
        lat_dim = lat_dims[0]
        bs = bss[0]
        vars = varss[0]
        noise = use_noises[0]
        label = "chm="
        filenames = [set_path + f'Comparison_{mode}/Dom_{dom_size[0]}/' + f_name + '_'.join([str(s) for s in steps[i]]) +\
                f"_dom_{dom_size[0]}_lat_{lat_dim}_bs_{bs}_chm_{chm}_var_{'_'.join(str(var) for var in vars)}_noise_{noise}" +\
                add_name + '_final.p' for i,chm in enumerate(chms)]
    elif mode=='var':
        lat_dim = lat_dims[0]
        bs = bss[0]
        chm = chms[0]
        noise = use_noises[0]
        label = "vars="
        filenames = [set_path + f'Comparison_{mode}/Dom_{dom_size[0]}/' + f_name + '_'.join([str(s) for s in steps[i]]) +\
                f"_dom_{dom_size[0]}_lat_{lat_dim}_bs_{bs}_chm_{chm}_var_{'_'.join(str(var) for var in vars)}_noise_{noise}" +\
                add_name + '_final.p' for i, vars in enumerate(varss)]
    elif mode=='noise':
        lat_dim = lat_dims[0]
        bs = bss[0]
        chm = chms[0]
        vars = varss[0]
        label = "noise="
        filenames = [set_path + f'Comparison_{mode}/Dom_{dom_size[0]}/' + f_name + '_'.join([str(s) for s in steps[i]]) +\
                f"_dom_{dom_size[0]}_lat_{lat_dim}_bs_{bs}_chm_{chm}_var_{'_'.join(str(var) for var in vars)}_noise_{use_noise}" +\
                add_name + '_final.p' for i,use_noise in enumerate(use_noises)]
    elif mode=='domain':
        lat_dim = lat_dims[0]
        bs = bss[0]
        chm = chms[0]
        vars = varss[0]
        noise = use_noises[0]
        label = "domain="
        filenames = [set_path + f'Comparison_{mode}/' + f_name + '_'.join([str(s) for s in steps[i]]) +\
                f"_dom_{dom_size[i]}_lat_{lat_dim}_bs_{bs}_chm_{chm}_var_{'_'.join(str(var) for var in vars)}_noise_{noise}" +\
                add_name + '_final.p' for i, _ in enumerate(dom_size)]
    else:
        print("something is wrong with your call")
        return 0

    for i, filename in enumerate(filenames):
        if not os.path.isfile(filename):
            #print("was not existant", mode)
            if mode=='lat':
                compute_final(set_path=set_path, steps=steps[i], f_name=f_name, add_name=add_name, mode=mode, 
                lat_dim=lat_dims[i], bs=bs, chm=chm, vars=vars, use_noise=noise, dom_size=dom_size[0], 
                no_mean=no_mean, mean_pert=mean_pert)
            if mode=='bs':
                compute_final(set_path=set_path, steps=steps[i], f_name=f_name, add_name=add_name, mode=mode, 
                lat_dim=lat_dim, bs=bss[i], chm=chm, vars=vars, use_noise=noise, dom_size=dom_size[0], 
                no_mean=no_mean, mean_pert=mean_pert)
            if mode=='chm':
                compute_final(set_path=set_path, steps=steps[i], f_name=f_name, add_name=add_name, mode=mode, 
                lat_dim=lat_dim, bs=bs, chm=chms[i], vars=vars, use_noise=noise, dom_size=dom_size[0], 
                no_mean=no_mean, mean_pert=mean_pert)
            if mode=='var':
                compute_final(set_path=set_path, steps=steps[i], f_name=f_name, add_name=add_name, mode=mode, 
                lat_dim=lat_dim, bs=bs, chm=chm, vars=varss[i], use_noise=noise, dom_size=dom_size[0], 
                no_mean=no_mean, mean_pert=mean_pert)
            if mode=='noise':
                compute_final(set_path=set_path, steps=steps[i], f_name=f_name, add_name=add_name, mode=mode, 
                lat_dim=lat_dim, bs=bs, chm=chm, vars=vars, use_noise=use_noises[i], dom_size=dom_size[0], 
                no_mean=no_mean, mean_pert=mean_pert)
            
            
    dics = [np.load(filename, allow_pickle=True) for filename in filenames]
    metrics = []
    for z in range(len(dics)):
        metrics = list(set(metrics).union(dics[z][steps[z][0]][0].keys()))
    metrics.sort()
    num_met = len(metrics)
    
    print("metrics: ", metrics)
    if 'SWD_metric' in dics[0][steps[0][0]][0].keys():
        num_met += 4
    names_psd = ['u','v','t2m']
    for z in range(len(varss)):
        names_psd = list(set(names_psd).union(set(varss[z])))

    num_met += len(names_psd)
    rows, cols = int(np.ceil(num_met/4)), 4
    fig = plt.figure(figsize=(cols*8,rows*6), facecolor='white')

    glob_ind = 1
    colors = ['red', 'green', 'blue', 'orange', 'grey', 'mediumorchid', 'deepskyblue']
    mapping_dic_for_spec_dist = {}
    domains = [f"{dom_size[i]}x{dom_size[i]}" for i in range(len(dom_size))]
    for i, metr in enumerate(metrics):
        if metr=='SWD_metric':
            for j in range(4):
                ax = plt.subplot(rows, cols, glob_ind)
                ax.set_title(f"SWD{2**(7-j)}*1e3", size=24)
                ax.set_xlabel('Steps' if x_label.lower()=='steps' else 'Images seen', size=18)
                if x_label.lower!='steps':
                    ax.set_xscale('log')
                ax.grid(visible=True)
                glob_ind += 1
                for k, dic in enumerate(dics):
                    means = []
                    stds = []
                    means.extend([dic[s]['Mean'][metr][j].item() for s in steps[k]])
                    stds.extend([dic[s]['Std'][metr][j].item() for s in steps[k]])
                    means = np.array(means)
                    stds = np.array(stds)
                    lab = lat_dims[k] if mode=='lat' else \
                          bss[k] if mode=='bs' else \
                          chms[k] if mode=='chm' else \
                          varss[k] if mode=='var' else \
                          use_noises[k] if mode=='noise' else\
                          domains[k]
                    ax.plot(x_values[k], means, color=colors[k], label=label+f'{lab}')
                    ax.fill_between(x_values[k], means+stds, means-stds, color=colors[k], alpha=0.2)
                ax.legend(framealpha = 0.8, fontsize=15)
            
            ax = plt.subplot(rows, cols, glob_ind)
            ax.set_title(f"SWD avg *1e3", size=24)
            ax.set_xlabel('Steps' if x_label.lower()=='steps' else 'Images seen', size=18)
            if x_label.lower!='steps':
                ax.set_xscale('log')
            ax.grid(visible=True)
            glob_ind += 1
            for k, dic in enumerate(dics):
                means = []
                stds = []
                means.extend([dic[s]['Mean'][metr][-1].item() for s in steps[k]])
                stds.extend([dic[s]['Std'][metr][-1].item() for s in steps[k]])
                means = np.array(means)
                stds = np.array(stds)
                lab = lat_dims[k] if mode=='lat' else \
                          bss[k] if mode=='bs' else \
                          chms[k] if mode=='chm' else \
                          varss[k] if mode=='var' else \
                          use_noises[k] if mode=='noise' else\
                          domains[k]
                ax.plot(x_values[k], means, color=colors[k], label=label+f'{lab}')
                ax.fill_between(x_values[k], means+stds, means-stds, color=colors[k], alpha=0.2)
            ax.legend(framealpha = 0.8, fontsize=12)

        elif 'spectral_dist' in metr:
            for k, dic in enumerate(dics):
                var_to_plot = varss[k] if mode=='var' else varss[0]
                for j in range(len(var_to_plot)):
                    if var_to_plot[j] in mapping_dic_for_spec_dist.keys():
                        ax = mapping_dic_for_spec_dist[var_to_plot[j]]
                    else:
                        ax = plt.subplot(rows, cols, glob_ind)
                        mapping_dic_for_spec_dist[var_to_plot[j]] = ax
                        glob_ind += 1
                    ax.set_title(f"PSD{var_to_plot[j]}", size=24)
                    ax.set_xlabel('Steps' if x_label.lower()=='steps' else 'Images seen', size=18)
                    ax.grid(visible=True)
                    if x_label.lower!='steps':
                        ax.set_xscale('log')
                    
                    means = []
                    stds = []
                    if metr in dic[steps[k][0]]['Mean'].keys():
                        if len(varss)==1:
                            means.extend([dic[s]['Mean'][metr][j].item() if len(var_to_plot)>1 \
                                        else dic[s]['Mean'][metr].item() for s in steps[k]])
                            stds.extend([dic[s]['Std'][metr][j].item() if len(var_to_plot)>1 \
                                      else dic[s]['Mean'][metr].item() for s in steps[k]])
                        else:
                            means.extend([dic[s]['Mean'][metr][j].item() if len(varss[k])>1 \
                                        else dic[s]['Mean'][metr].item() for s in steps[k]])
                            stds.extend([dic[s]['Std'][metr][j].item() if len(varss[k])>1 \
                                      else dic[s]['Mean'][metr].item() for s in steps[k]])
                        means = np.array(means)
                        stds = np.array(stds)
                        lab = lat_dims[k] if mode=='lat' else \
                            bss[k] if mode=='bs' else \
                            chms[k] if mode=='chm' else \
                            varss[k] if mode=='var' else \
                            use_noises[k] if mode=='noise' else\
                            domains[k]
                        #print(colors[k], varss[k])
                        ax.plot(x_values[k], means, color=colors[k], label=label+f'{lab}')
                        ax.fill_between(x_values[k], means+stds, means-stds, color=colors[k], alpha=0.2)
                    ax.legend(framealpha = 0.8, fontsize=12)
        elif metr=='multivar':
            ax = plt.subplot(rows, cols, glob_ind)
            ax.set_title("KL divergence", size=24)
            ax.set_xlabel('Steps' if x_label.lower()=='steps' else 'Images seen', size=18)
            if x_label.lower!='steps':
                ax.set_xscale('log')
            ax.grid(visible=True)
            glob_ind += 1
            for j, dic in enumerate(dics):
                KL_div_means, KL_div_stds = [], []
                for s in steps[j]:
                    corr_r_mean, corr_f_mean = dic[s]['Mean'][metr][0]*(2/101)**2, dic[s]['Mean'][metr][1]*(2/101)**2
                    corr_r_std, corr_f_std = dic[s]['Std'][metr][0]*(2/101)**2, dic[s]['Std'][metr][1]*(2/101)**2
                    
                    idx_mean = np.logical_and(corr_r_mean!=0, corr_f_mean!=0)
                    idx_std = np.logical_and(corr_r_std!=0, corr_f_std!=0)

                    kl_div_mean = np.sum(corr_r_mean[idx_mean]*np.log(corr_r_mean[idx_mean]/corr_f_mean[idx_mean]))
                    kl_div_std = np.sum(corr_r_std[idx_std]*np.log(corr_r_std[idx_std]/corr_f_std[idx_std]))
                    KL_div_means.append(kl_div_mean)
                    KL_div_stds.append(kl_div_std)

                KL_div_means, KL_div_stds = np.array(KL_div_means), np.array(KL_div_stds)

                lab = lat_dims[j] if mode=='lat' else \
                          bss[j] if mode=='bs' else \
                          chms[j] if mode=='chm' else \
                          varss[j] if mode=='var' else \
                          use_noises[j] if mode=='noise' else\
                          domains[j]
                ax.plot(x_values[j], KL_div_means, color=colors[j], label=label+f'{lab}')
                ax.fill_between(x_values[j], KL_div_means+KL_div_stds, KL_div_means-KL_div_stds, color=colors[j], alpha=0.2)
            ax.legend(framealpha = 0.8, fontsize=12)
        else:
            ax = plt.subplot(rows, cols, glob_ind)
            ax.set_title(metr, size=24)
            ax.set_xlabel('Steps' if x_label.lower()=='steps' else 'Images seen', size=18)
            if x_label.lower!='steps':
                ax.set_xscale('log')
            ax.grid(visible=True)
            glob_ind += 1
            for j, dic in enumerate(dics):
                means = []
                stds = []
                means.extend([dic[s]['Mean'][metr].item() for s in steps[j]])
                stds.extend([dic[s]['Std'][metr].item() for s in steps[j]])
                means = np.array(means)
                stds = np.array(stds)
                lab = lat_dims[j] if mode=='lat' else \
                          bss[j] if mode=='bs' else \
                          chms[j] if mode=='chm' else \
                          varss[j] if mode=='var' else \
                          use_noises[j] if mode=='noise' else\
                          domains[j]
                ax.plot(x_values[j], means, color=colors[j], label=label+f'{lab}')
                ax.fill_between(x_values[j], means+stds, means-stds, color=colors[j], alpha=0.2)
            ax.legend(framealpha = 0.8, fontsize=12)
    suptitle = f"Comparison of different distance metrics (domain = {dom_size[0]}x{dom_size[0]})\nfor several "
    save_path = set_path + f"Comparison_{mode}/" + (f"Dom_{dom_size[0]}/" if len(dom_size)==1 else "")
    if mode=='lat':
        save_name = save_path + "compare_metr_lat_dims_" + '_'.join(str(lat_dim) for\
            lat_dim in lat_dims) + f'_dom_{dom_size[0]}_steps_' + '_'.join(str(step) for step in steps[0]) + f'_bs_{bs}_chm_{chm}'
        suptitle += f"latent dimensions"
    if mode=='bs':
        save_name = save_path + "compare_metr_bss_" + '_'.join(str(bs) for bs in bss) +\
            f'_dom_{dom_size[0]}_steps_' + '_'.join(str(step) for step in steps[0]) + f'_lat_{lat_dim}_chm_{chm}'
        save_name += "_wrt_imgs" if x_label.lower()!='steps' else ""
        suptitle += f"batch sizes"
    if mode=='chm':
        save_name = save_path + "compare_metr_chms_" + '_'.join(str(chm) for chm in chms) + f'_dom_{dom_size[0]}_steps_' + \
                '_'.join(str(step) for step in steps[0]) + f'_bs_{bs}_lat_{lat_dim}'
        suptitle += f"channel multipliers"
    if mode=='var':
        save_name = save_path + "compare_metr_vars_" + '|'.join('_'.join(str(var) for var in vars) for vars in varss)\
                        + f'_dom_{dom_size[0]}_steps_' + '_'.join(str(step) for step in steps[0]) + f'_bs_{bs}_lat_{lat_dim}_chm_{chm}'
        suptitle += f"vars"
    if mode=='noise':
        save_name = save_path + "compare_metr_noise_" + '_'.join(str(var) for var in vars)\
                        + f'_dom_{dom_size[0]}_steps_' + '_'.join(str(step) for step in steps[0]) + f'_bs_{bs}_lat_{lat_dim}_chm_{chm}'
        suptitle += f"noise configs"
    if mode=='domain':
        save_name = save_path + f"compare_metr_domains_{'_'.join(str(dom_s) for dom_s in dom_size)}_" +\
        '_'.join(str(var) for var in vars) + '_steps_' + '_'.join(str(step) for step in steps[0]) + f'_bs_{bs}_lat_{lat_dim}_chm_{chm}'
        suptitle += f"domain sizes"
    save_name += "_mean_pert" if mean_pert else "_no_mean" if no_mean else ""
   
    fig.suptitle(suptitle, fontsize=35)
    fig.subplots_adjust(bottom=0.005, top=0.88, left=0.05, right=0.95, wspace=0.1, hspace=0.25)
    fig.savefig(save_name + add_name + '.png')



def compare_swd_decoupled(set_path, steps=[[51000,102000,147000]], add_name='_16384_16384', lat_dims=[512], chms=[2], bss=[16],
                varss=[['u','v','t2m']], use_noises=[True], dom_size=[128], no_mean=False, mean_pert=False, x_label='steps'):
    
    # x_label: plot with respect to nb of iterations ('steps') or images seen ('images' or anything else)
    # for now works only with mode='bs' because not relevant otherwise

    filenames = []
    f_name = "test_for_score_crawl_swd_decoupled_distance_metrics_step_"
    dic_len = {'noise':len(use_noises),'var': len(varss),'lat':len(lat_dims), 'bs':len(bss), 'chm':len(chms)}
    mode = max(dic_len, key=dic_len.get)
    print(mode)

    set_path += f"Finished_experiments/"
    if len(steps)==1:
        steps = [steps[0] for i in range(dic_len[mode])]
    x_values = steps
    
    if mode=='lat':
        bs = bss[0]
        chm = chms[0]
        vars = varss[0]
        label = "Lat_dim="
        filenames = [set_path + f'Comparison_{mode}/Dom_{dom_size[0]}/' + f_name + '_'.join([str(s) for s in steps[i]]) +\
                f"_dom_{dom_size[0]}_lat_{lat_dim}_bs_{bs}_chm_{chm}_var_{'_'.join(str(var) for var in vars)}" +\
                add_name + '_final.p' for i,lat_dim in enumerate(lat_dims)]
    elif mode=='bs':
        lat_dim = lat_dims[0]
        chm = chms[0]
        vars = varss[0]
        noise = use_noises[0]
        label= "Batch size="
        filenames = [set_path + f'Comparison_{mode}/Dom_{dom_size[0]}/' + f_name + '_'.join([str(s) for s in steps[i]]) +\
                f"_dom_{dom_size[0]}_lat_{lat_dim}_bs_{bs}_chm_{chm}_var_{'_'.join(str(var) for var in vars)}" +\
                add_name + '_final.p' for i,bs in enumerate(bss)]
        if not x_label.lower()=='steps':
            x_values = [np.array(steps[i])*bss[i] for i in range(len(bss))]            
    elif mode=='chm':
        lat_dim = lat_dims[0]
        bs = bss[0]
        vars = varss[0]
        noise = use_noises[0]
        label = "chm="
        filenames = [set_path + f'Comparison_{mode}/Dom_{dom_size[0]}/' + f_name + '_'.join([str(s) for s in steps[i]]) +\
                f"_dom_{dom_size[0]}_lat_{lat_dim}_bs_{bs}_chm_{chm}_var_{'_'.join(str(var) for var in vars)}" +\
                add_name + '_final.p' for i,chm in enumerate(chms)]
    elif mode=='var':
        lat_dim = lat_dims[0]
        bs = bss[0]
        chm = chms[0]
        noise = use_noises[0]
        label = "vars="
        filenames = [set_path + f'Comparison_{mode}/Dom_{dom_size[0]}/' + f_name + '_'.join([str(s) for s in steps[i]]) +\
                f"_dom_{dom_size[0]}_lat_{lat_dim}_bs_{bs}_chm_{chm}_var_{'_'.join(str(var) for var in vars)}" +\
                add_name + '_final.p' for i, vars in enumerate(varss)]
    elif mode=='noise':
        lat_dim = lat_dims[0]
        bs = bss[0]
        chm = chms[0]
        vars = varss[0]
        label = "noise="
        filenames = [set_path + f'Comparison_{mode}/Dom_{dom_size[0]}/' + f_name + '_'.join([str(s) for s in steps[i]]) +\
                f"_dom_{dom_size[0]}_lat_{lat_dim}_bs_{bs}_chm_{chm}_var_{'_'.join(str(var) for var in vars)}_noise_{use_noise}" +\
                add_name + '_final.p' for i,use_noise in enumerate(use_noises)]
    else:
        print("something is wrong with your call")
        return 0
    
    
    for i, fname in enumerate(filenames):
        #print(fname)
        if not os.path.isfile(fname):
            if mode=='lat':
                compute_final(set_path=set_path, steps=steps[i], f_name=f_name, add_name=add_name, mode=mode, 
                lat_dim=lat_dims[i], bs=bs, chm=chm, vars=vars, use_noise=noise, dom_size=dom_size[0], 
                no_mean=no_mean, mean_pert=mean_pert)
            if mode=='bs':
                compute_final(set_path=set_path, steps=steps[i], f_name=f_name, add_name=add_name, mode=mode, 
                lat_dim=lat_dim, bs=bss[i], chm=chm, vars=vars, use_noise=noise, dom_size=dom_size[0], 
                no_mean=no_mean, mean_pert=mean_pert)
            if mode=='chm':
                compute_final(set_path=set_path, steps=steps[i], f_name=f_name, add_name=add_name, mode=mode, 
                lat_dim=lat_dim, bs=bs, chm=chms[i], vars=vars, use_noise=noise, dom_size=dom_size[0], 
                no_mean=no_mean, mean_pert=mean_pert)
            if mode=='var':
                compute_final(set_path=set_path, steps=steps[i], f_name=f_name, add_name=add_name, mode=mode, 
                lat_dim=lat_dim, bs=bs, chm=chm, vars=varss[i], use_noise=noise, dom_size=dom_size[0], 
                no_mean=no_mean, mean_pert=mean_pert)
            if mode=='noise':
                compute_final(set_path=set_path, steps=steps[i], f_name=f_name, add_name=add_name, mode=mode, 
                lat_dim=lat_dim, bs=bs, chm=chm, vars=vars, use_noise=use_noises[i], dom_size=dom_size[0], 
                no_mean=no_mean, mean_pert=mean_pert)
            
    dics = [np.load(filename, allow_pickle=True) for filename in filenames]
    metrics = []
    all_var = []
    if mode!='var':
        all_var = varss[0]
    else:
        all_var = list(set(flatten(varss)))

    metrics = [f"SWD_metric_{str(var)}" for var in all_var]    
    metrics.sort()
    num_met = len(metrics)*5 

    print("metrics: ", metrics)
    
    rows, cols = int(np.ceil(num_met/5)), 5
    fig = plt.figure(figsize=(cols*6,rows*5), facecolor='white')

    glob_ind = 1
    colors = ['red', 'green', 'blue', 'orange', 'grey', 'mediumorchid', 'deepskyblue']
    for i, metr in enumerate(metrics):
        for j in range(4):
            ax = plt.subplot(rows, cols, glob_ind)
            ax.set_title(f"SWD{2**(7-j)}*1e3 - {metr[11:]}", size=24)
            ax.set_xlabel('Steps' if x_label.lower()=='steps' else 'Images seen', size=18)
            if x_label.lower!='steps':
                ax.set_xscale('log')
            ax.grid(visible=True)
            glob_ind += 1
            if mode=='var':
                for k, vars in enumerate(varss):
                    if sum(v in metr for v in vars)>=1:
                        means = []
                        stds = []
                        means.extend([dics[k][s]['Mean'][metr][j] for s in steps[k]])
                        stds.extend([dics[k][s]['Std'][metr][j] for s in steps[k]])
                        means = np.array(means)
                        stds = np.array(stds)
                        lab = lat_dims[k] if mode=='lat' else \
                            bss[k] if mode=='bs' else \
                            chms[k] if mode=='chm' else \
                            varss[k] if mode=='var' else \
                            use_noises[k] if mode=='noise' else\
                            ""
                        ax.plot(x_values[k], means, color=colors[k], label=label+f'{lab}')
                        ax.fill_between(x_values[k], means+stds, means-stds, color=colors[k], alpha=0.2)
            else:
                for k, dic in enumerate(dics):
                    if sum(v in metr for v in vars)>=1:
                        means = []
                        stds = []
                        means.extend([dic[s]['Mean'][metr][j] for s in steps[k]])
                        stds.extend([dic[s]['Std'][metr][j] for s in steps[k]])
                        means = np.array(means)
                        stds = np.array(stds)
                        lab = lat_dims[k] if mode=='lat' else \
                            bss[k] if mode=='bs' else \
                            chms[k] if mode=='chm' else \
                            varss[k] if mode=='var' else \
                            use_noises[k] if mode=='noise' else\
                            ""
                        ax.plot(x_values[k], means, color=colors[k], label=label+f'{lab}')
                        ax.fill_between(x_values[k], means+stds, means-stds, color=colors[k], alpha=0.2)
            ax.legend(framealpha = 0.8, fontsize=15)
            
        ax = plt.subplot(rows, cols, glob_ind)
        ax.set_title(f"SWD avg *1e3 - {metr[11:]}", size=24)
        ax.set_xlabel('Steps' if x_label.lower()=='steps' else 'Images seen', size=18)
        if x_label.lower!='steps':
            ax.set_xscale('log')
        ax.grid(visible=True)
        glob_ind += 1
        if mode=='var':
            for k, vars in enumerate(varss):
                if sum(v in metr for v in vars)>=1:
                    means = []
                    stds = []
                    means.extend([dics[k][s]['Mean'][metr][-1] for s in steps[k]])
                    stds.extend([dics[k][s]['Std'][metr][-1] for s in steps[k]])
                    means = np.array(means)
                    stds = np.array(stds)
                    lab = lat_dims[k] if mode=='lat' else \
                            bss[k] if mode=='bs' else \
                            chms[k] if mode=='chm' else \
                            varss[k] if mode=='var' else \
                            use_noises[k] if mode=='noise' else\
                            ""
                    ax.plot(x_values[k], means, color=colors[k], label=label+f'{lab}')
                    ax.fill_between(x_values[k], means+stds, means-stds, color=colors[k], alpha=0.2)
        else:
            for k, dic in enumerate(dics):
                if sum(v in metr for v in vars)>=1:
                        means = []
                        stds = []
                        means.extend([dic[s]['Mean'][metr][-1] for s in steps[k]])
                        stds.extend([dic[s]['Std'][metr][-1] for s in steps[k]])
                        means = np.array(means)
                        stds = np.array(stds)
                        lab = lat_dims[k] if mode=='lat' else \
                            bss[k] if mode=='bs' else \
                            chms[k] if mode=='chm' else \
                            varss[k] if mode=='var' else \
                            use_noises[k] if mode=='noise' else\
                            ""
                        ax.plot(x_values[k], means, color=colors[k], label=label+f'{lab}')
                        ax.fill_between(x_values[k], means+stds, means-stds, color=colors[k], alpha=0.2)

        ax.legend(framealpha = 0.8, fontsize=12)
    
    suptitle = f"Comparison of decoupled SWD (domain = {dom_size[0]}x{dom_size[0]})\nfor several "
    save_path = set_path + f"Comparison_{mode}/Dom_{dom_size[0]}/"
    if mode=='lat':
        save_name = save_path + "compare_swd_decoup_lat_dims_" + '_'.join(str(lat_dim) for\
            lat_dim in lat_dims) + f'dom_{dom_size[0]}_steps_' + '_'.join(str(step) for step in steps[0]) + f'_bs_{bs}_chm_{chm}'
        suptitle += f"latent dimensions"
    if mode=='bs':
        save_name = save_path + "compare_swd_decoup_bss_" + '_'.join(str(bs) for bs in bss) +\
            f'dom_{dom_size[0]}_steps_' + '_'.join(str(step) for step in steps[0]) + f'_lat_{lat_dim}_chm_{chm}'
        save_name += "_wrt_imgs" if x_label.lower()!='steps' else ""
        suptitle += f"batch sizes"
    if mode=='chm':
        save_name = save_path + "compare_swd_decoup_chms_" + '_'.join(str(chm) for chm in chms) + '_steps_' + \
                '_'.join(str(step) for step in steps[0]) + f'_bs_{bs}_lat_{lat_dim}'
        suptitle += f"channel multipliers"
    if mode=='var':
        save_name = save_path + "compare_swd_decoup_vars_" + '|'.join('_'.join(str(var) for var in vars) for vars in varss)\
                    + f'dom_{dom_size[0]}_steps_' + '_'.join(str(step) for step in steps[0]) + f'_bs_{bs}_lat_{lat_dim}_chm_{chm}'
        suptitle += f"vars"
    if mode=='noise':
        save_name = save_path + "compare_swd_decoup_noise_" + '_'.join(str(var) for var in vars)\
                    + f'dom_{dom_size[0]}_steps_' + '_'.join(str(step) for step in steps[0]) + f'_bs_{bs}_lat_{lat_dim}_chm_{chm}'
        suptitle += f"noise configs"
    save_name += "_mean_pert" if mean_pert else "_no_mean" if no_mean else ""


    fig.suptitle(suptitle, fontsize=35)
    fig.subplots_adjust(bottom=0.005, top=0.88, left=0.05, right=0.95, wspace=0.1, hspace=0.3)
    fig.savefig(save_name + add_name + '.png', dpi=400)

def plot_metr_histo(lats=[512], bss=[16], chms=[2], varss=[['u','v','t2m']], no_mean=False, mean_pert=False,
                 noises=[True], dom_size=[128], add_name='_16384_16384', metr="SWD_metric", ind_baseline=3,
                 exp_path="/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/Finished_experiments/"):
    

    dic_len = {'noise':len(noises),'var': len(varss),'lat':len(lats), 'bs':len(bss), 'chm':len(chms)}
    mode = max(dic_len, key=dic_len.get)
    print(mode)

    if metr=="SWD_metric":
        metr_name = "SWD avg"
    else:
        metr_name = metr
    save_path = exp_path + f"Comparison_{mode}/Dom_{dom_size[0]}/"

    nomean, mean_p = "_no_mean" if no_mean else "", "_mean_pert" if mean_pert else ""

    suptitle = f"Values of {metr_name} after convergence for different "
    save_name = f"histo_{metr}_{dom_size[0]}_var_{'_'.join(var for var in varss[0])}_noise_"\
        + f"{'_'.join(str(noise) for noise in noises)}_lat_{'_'.join(str(lat) for lat in lats)}"\
        + f"_bs_{'_'.join(str(bs) for bs in bss)}_ch_{'_'.join(str(ch) for ch in chms)}{nomean}{mean_p}.png"
    
    if mode=='lat':
        suptitle += f"latent dimensions"
        tick_labels = lats
        filenames = [glob(save_path+f"test_for_score_crawl_dis*lat_{lat}_bs_{bss[0]}_chm_{chms[0]}_var_"+\
                f"{'_'.join(var for var in varss[0])}{add_name}_final.p")[0] for lat in lats]
    if mode=='bs':
        suptitle += f"batch sizes"
        tick_labels = bss
        filenames = [glob(save_path+f"test_for_score_crawl_dis*lat_{lats[0]}_bs_{bs}_chm_{chms[0]}_var_"+\
                f"{'_'.join(var for var in varss[0])}{add_name}_final.p")[0] for bs in bss]
    if mode=='chm':
        suptitle += f"channel multipliers"
        tick_labels = chms
        filenames = [glob(save_path+f"test_for_score_crawl_dis*lat_{lats[0]}_bs_{bss[0]}_chm_{chm}_var_"+\
                f"{'_'.join(var for var in varss[0])}{add_name}_final.p")[0] for chm in chms]
    if mode=='var':
        suptitle += f"combinations of variables"
        tick_labels = varss
        filenames = [glob(save_path+f"test_for_score_crawl_dis*lat_{lats[0]}_bs_{bss[0]}_chm_{chms[0]}_var_"+\
                f"{'_'.join(var for var in vars)}{add_name}_final.p")[0] for vars in varss]
        save_name = f"histo_{metr}_{dom_size[0]}_"
        save_name += f"var_{'|'.join('_'.join(var for var in vars) for vars in varss)}_noise_{'_'.join(str(noise) for noise in noises)}_"\
            + f"lat_{'_'.join(str(lat) for lat in lats)}_bs_{'_'.join(str(bs) for bs in bss)}_ch_{'_'.join(str(ch) for ch in chms)}.png"
    if mode=='noise':
        suptitle += f"noise configurations"
        tick_labels = noises
        filenames = [glob(save_path+f"test_for_score_crawl_dis*lat_{lats[0]}_bs_{bss[0]}_chm_{chms[0]}_var_"+\
                f"{'_'.join(var for var in varss[0])}_noise_{noise}{add_name}_final.p")[0] for noise in noises]

    suptitle += f"\nDomain size: {dom_size[0]}x{dom_size[0]}"
    filenames = flatten(filenames)
    files = [np.load(filename, allow_pickle=True) for filename in filenames]

    keys = [list(file.keys()) for file in files]
    print([keys[i][-1] for i in range(len(keys))])

    Means = [file[keys[i][-1]]["Mean"][metr][-1].item() for i, file in enumerate(files)]
    Stds = [file[keys[i][-1]]["Std"][metr][-1].item() for i, file in enumerate(files)]

    height = np.array(Means)
    yerr = np.array(Stds)
    x_pos = np.linspace(0,1,len(Means))
    colors = ['red', 'green', 'blue', 'orange', 'grey', 'yellow', 'deepskyblue']
    #values_text = [f"{round(Means[i],4)}\n" + r'$\pm$' + f"\n{round(Stds[i],4)}" for i in range(len(Means))]

    tick_labels = [multiline_label(label) for label in tick_labels]
    fig, ax = plt.subplots(1,1,figsize=(8,6), facecolor='white')

    bar_plot = ax.bar(x=x_pos, height=height, yerr=yerr, color=colors[:len(x_pos)], capsize=15*5/len(Means), width=1/len(Means))
    ax.set_xticks(x_pos, labels=tick_labels, rotation=45 if mode=='var' else 0, fontsize=20 if mode!="var" else 10, 
                    rotation_mode="anchor", ha="right" if mode=='var' else "center", ma='center')
    ax.set_ylabel(metr_name, fontsize=20)
    ax.set_title(suptitle, fontsize=15)
    ax.bar_label(bar_plot, labels=[""]*(ind_baseline-1)+["Baseline"]+[""]*(height.shape[0]-ind_baseline), fontsize=15)
    #for bar, value_text in zip(ax.patches, values_text[::1]):
    #    ax.text(bar.get_x()+bar.get_width()/2, bar.get_y()+bar.get_height()/2, value_text, color = 'black', 
    #            ha = 'left', va = 'center', ma = 'center') 

    fig.savefig(save_path+save_name, dpi=400, bbox_inches="tight")



def plot_weight_noise_injection(lat=512, bs=16, chm=2, vars=['u','v','t2m'], 
                                dom_size=128, jump=8, ceil=1e-4, no_mean=False, mean_pert=False,
                                model_path="/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/Finished_experiments/",
                                savePath="/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/weights_plot/"):


    nomean, mean_p = "_no_mean" if no_mean else "", "_mean_pert" if mean_pert else ""
    
    def record_noises(lat, bs, chm, vars, dom_size, ceil, no_mean, mean_pert, ckpt_start, ckpt_end, ckpt_space, ckpt_list):
        """
        record all weights of noise injection bigger than ceil in a file for all checkpoints
        for easier use afterwards
        """

        nomean, mean_p = "_no_mean" if no_mean else "", "_mean_pert" if mean_pert else ""
        recorded_ckpt_names = f"{ckpt_start}_to_{ckpt_end}_every_{ckpt_space}"
        save_path = savePath + f"weights_dom_{dom_size}_vars_{'_'.join(var for var in vars)}"\
            + f"_lat_{lat}_bs_{bs}_chm_{chm}_{recorded_ckpt_names}{nomean}{mean_p}_ceil_{ceil}.npy"
        
        weights = np.zeros((len(ckpt_list),nb_weights))
        for i, ckpt_n in enumerate(ckpt_list):
            if (i+1)%4==0:
                print(f"ckpt {i+1}/{len(ckpt_list)}")
            dic = torch.load(ckpt_n, map_location=torch.device('cpu'))['g_ema']
            for j in range(nb_weights):
                weights[i,j] = dic[f"convs.{j}.noise.weight"].item()
        np.save(save_path, weights)

    nb_weights = (int(math.log(dom_size,2))-2)*2
    model_name = model_path + f"stylegan2_stylegan_dom_{dom_size}_lat-dim_{lat}_bs_{bs}"+\
                f"_0.002_0.002_ch-mul_{chm}_vars_" +\
                f"{'_'.join(var for var in vars)}_noise_True" +\
                f"{nomean}{mean_p}/Instance_1/models/"
    
    ckpt_names = glob(model_name + "*")
    ckpt_names.sort()
    ckpt_for_plot = ckpt_names if len(ckpt_names)<20 else ckpt_names[::2] # reduce a bit if too many ckpts
    nb_ckpt = len(ckpt_for_plot)
    print("nb ckpt: ", nb_ckpt)

    x_labels = [(int(ckpt_n[-9:-3].lstrip('0')) if ckpt_n[-9:-3]!='000000' else int(ckpt_n[-9:-3]))  for ckpt_n in ckpt_for_plot]
    x_labels.sort()

    ckpt_start, ckpt_end, ckpt_space = x_labels[0], x_labels[-1], x_labels[1]-x_labels[0]

    recorded_ckpt_names = f"{ckpt_start}_to_{ckpt_end}_every_{ckpt_space}"
    f_name_weights = savePath + f"weights_dom_{dom_size}_vars_{'_'.join(var for var in vars)}"\
            + f"_lat_{lat}_bs_{bs}_chm_{chm}_{recorded_ckpt_names}{nomean}{mean_p}_ceil_{ceil}.npy"
    
    if not os.path.isfile(f_name_weights):
        record_noises(lat, bs, chm, vars, dom_size, ceil, no_mean, mean_pert, ckpt_start, ckpt_end, ckpt_space)
    
    fig, ax = plt.subplots(1,1,figsize=(8,6), facecolor='white')
    
    weights = np.load(f_name_weights)
    for i in range(nb_weights):
        # only plot if a majority along the training is above the ceil
        if np.count_nonzero(np.abs(weights[:,i])[1:]>ceil)>weights[:,i][1:].shape[0]//2:
            ax.plot(x_labels, np.abs(weights[:,i]), label=f"|weight {i}|")
    ax.legend()
    ax.set_yscale('log')
    ax.set_title(f"Abs values of the noise injection weights during the training if bigger than {ceil}")
    picname = savePath + f"img_dom_{dom_size}_vars_{'_'.join(var for var in vars)}_lat_{lat}_bs_{bs}_chm_{chm}_"+\
        f"{ckpt_start}_to_{ckpt_end}_every_{ckpt_space}_ceil_{ceil}.png"
    fig.savefig(picname, dpi=400)



def plot_ls_corr(lat, bs, chm, ckpt, out_dir, lr=0.002, cmap_wind='viridis', dom_size=128, no_mean=False, mean_pert=False,
                cmap_t='rainbow', vars=['u','v','t2m'], use_noise=True, N=16384, 
                exp_path="/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/Finished_experiments/",
                datapath="/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/",
                home_path="/home/poulainauzeaul/home-priam-sidev/"):
    from projection_artistic import artistic as art

    assert not (no_mean and mean_pert), "You can't have no_mean and mean_pert activated at the same time"


    data_dir = datapath + ("IS_1_1.0_0_0_0_0_0_256_mean_0/" if no_mean \
                             else "IS_1_1.0_0_0_0_0_0_256_mean_pert/" if mean_pert \
                             else "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/")

    fsamp_dir = exp_path + "Finished_experiments/" +\
            f"stylegan2_stylegan_dom_{dom_size}_lat-dim_{lat}_bs_{bs}_{lr}_{lr}_ch-mul_{chm}"
    fsamp_dir += f"_vars_{'_'.join(var for var in vars)}_noise_{use_noise}"
    fsamp_dir += "_no_mean" if no_mean else ""
    fsamp_dir += "_mean_pert" if mean_pert else ""
    fsamp_dir += "/Instance_1/log/"
    filename = f"test_for_score_crawl_standalone_metrics_step_{ckpt}_{N}.p"

    var_arome = ['u','v','t2m','z500','t850','tpw850']
    var_dict_plot = {var: i for i, var in enumerate(sorted(vars, key=lambda var_:var_dict[var_]))}
    arome_metr_dic = np.load(datapath+"/test_for_score_crawl_standalone_metrics"+\
                             f"_{'_'.join(var for var in var_arome)}_dom_{dom_size}_{N}.p", allow_pickle=True)
    data_arome = arome_metr_dic["ls_metric"][[var_dict_plot[var] for var in vars]]
    arome_corr = data_arome.reshape(-1, *data_arome.shape)

    dic = np.load(fsamp_dir+filename, allow_pickle=True)

    if dic["ls_metric"].ndim==2: # only one var
        data = dic["ls_metric"].reshape((1,1,*dic["ls_metric"].shape))
    elif dic["ls_metric"].ndim==3: # we have several vars
        data = dic["ls_metric"].reshape(-1,*dic["ls_metric"].shape)
    
    data = np.concatenate([arome_corr, data], axis=0)

    pic_name = f"L_corr_" + f"_sample_dom_{dom_size}_lat_{lat}_bs_{bs}_"+\
                f"chm_{chm}_lr_{lr}_ckpt_{ckpt}_{'_'.join(var for var in vars)}_noise_{use_noise}"
    pic_name += "_no_mean.png" if no_mean else "_mean_pert.png" if mean_pert else ".png"
    col_titles = ["AROME"] + ["GAN"]

    suptitle = f"L_corr distances for different variables"
    
    zone = "SE_for_GAN" if dom_size==128 else "SE_GAN_extend" if dom_size==256 else "AROME_all"
    can = art.canvasHolder(zone, nb_lon=dom_size, nb_lat=np.min([dom_size, 717]), fpath=home_path)
    var_names = [(var, "km") for var in vars]
    can.plot_abs_error(data=data, var_names=var_names, suptitle=suptitle,
                        plot_dir=out_dir, pic_name=pic_name, col_titles=col_titles, cmap_wind=cmap_wind, cmap_t=cmap_t)
    # a voir mais le mieux semble de mettre une cbar commune et arome v gan seulement



def load_rgbs(program, lat=512, bs=16, chm=2, lr=0.002, ckpt=147000, vars=['u','v','t2m'], 
                rgb_level=3, use_noise=True, dom_size=128, no_mean=False, mean_pert=False,
                rgb_path="/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/Finished_experiments/"):
    
    assert not (no_mean and mean_pert), "You can't have no_mean and mean_pert activated at the same time"


    data_dir = rgb_path\
                + f"stylegan2_stylegan_dom_{dom_size}_lat-dim_{lat}_bs_{bs}_{lr}_{lr}_ch-mul_{chm}"
    data_dir += f"_vars_{'_'.join(var for var in vars)}_noise_{use_noise}" + ("_no_mean" if no_mean else "")
    data_dir += "_mean_pert" if mean_pert else ""
    data_dir += "/Instance_1/toRGB_outs/"
    print(data_dir)
    name = f"RGBS_level_{rgb_level}_lat_{lat}_bs_{bs}_chm_{chm}_lr_{lr}_ckpt_{ckpt}_"
    print("nb files: ", len(glob(data_dir+name+"*")))
    glob_list = glob(data_dir+name+"*")

    res = {}

    for key, value in program.items():
        if value[0]==1:
            
            fileList=random.sample(glob_list,value[1]//256 if dom_size==128 else value[1]//512)
            
            res[key]=fileList
    
    return res, data_dir


def compute_normed_rgb(program, lat=512, bs=16, chm=2, lr=0.002, ckpt=147000, vars=['u','v','t2m'], dom_size=128,
        rgb_level=3, use_noise=True, norma='classic', no_mean=False, mean_pert=False, 
        rgb_path="/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/Finished_experiments/",
        datapath="/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/"):

    assert not (no_mean and mean_pert), "You can't have no_mean and mean_pert activated at the same time"
    # classic norma: with Means and Stds, otherwise with spatial std
    
    upsampling_mode='nearest'

    if norma.lower()!='classic':
        norma='std'
    
    data_dir_0 = datapath + ("IS_1_1.0_0_0_0_0_0_256_mean_0/" if no_mean \
                             else "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/")
    
    data_dir_mean_pert = datapath + "IS_1_1.0_0_0_0_0_0_256_mean_pert/"
    
    mode_name_0 = "_with_0_mean" if no_mean else "_with_8_var"
    mode_name_mean_pert = "_mean_pert"
    
    means_fname_0 = f'mean{mode_name_0}.npy'
    maxs_fname_0 = f'max{mode_name_0}.npy'
    means_fname_mean_pert = f'mean{mode_name_mean_pert}.npy'
    maxs_fname_mean_pert = f'max{mode_name_mean_pert}.npy'

    Means = np.load(data_dir_0 + means_fname_0)[[var_dict[var] for var in vars]].reshape(len(vars),1,1)
    Maxs = np.load(data_dir_0 + maxs_fname_0)[[var_dict[var] for var in vars]].reshape(len(vars),1,1)
    
    if mean_pert:
        # we will need to denorm before renormalizing
        var_idx = [var_dict[var] for var in vars]
        var_idx += [vi+8 for vi in var_idx]
        Means2 = np.load(data_dir_mean_pert + means_fname_mean_pert)[[var_id for var_id in var_idx]].reshape(len(var_idx),1,1)
        Maxs2 = np.load(data_dir_mean_pert + maxs_fname_mean_pert)[[var_id for var_id in var_idx]].reshape(len(var_idx),1,1)
        Stds2 = (1.0/0.95) * Maxs2

    Stds = (1.0/0.95) * Maxs

    res, rgb_folder_path = load_rgbs(program=program, lat=lat, bs=bs, chm=chm, lr=lr, ckpt=ckpt, dom_size=dom_size,
                vars=vars, rgb_level=rgb_level, use_noise=use_noise, no_mean=no_mean, mean_pert=mean_pert, rgb_path=rgb_path)
    N = program[0][1]
    biglist = [np.load(res[0][i], allow_pickle=True).item() for i in range(len(res[0]))]
    print("data loaded")

    nb_res = len(biglist[0])
    nb_var = len(vars)
    max_res = dom_size
    size_str = f"{max_res}x{max_res}"
    for i in range(1, nb_res+1):
        print(f"res: {2**(i+1)}")
        upsampler = torch.nn.Upsample(size=max_res, mode=upsampling_mode)
        concat = np.concatenate([upsampler(torch.from_numpy(biglist[j][i])).numpy() for j in range(len(biglist))])
        if norma.lower()=='classic':
            if not mean_pert:
                concat_normalised = np.copy(concat) # nothing to do here
            else:
                concat_tmp = concat * Stds2 + Means2
                concat_normalised = np.zeros((concat_tmp.shape[0], len(vars), max_res, max_res))
                for j in range(concat_normalised.shape[1]):
                    concat_normalised[:,j,:,:] = concat_tmp[:,j,:,:] + concat_tmp[:,j+len(vars),:,:]
                
                concat_normalised = (concat_normalised-Means)/Stds
                
        else:
            std_all_but_var = np.std(concat, axis=(0, 2, 3)).reshape(1,nb_var,1,1)
            concat_normalised = (concat*Stds+Means)/std_all_but_var

        name = f"rgb_lvl_{rgb_level}_{size_str}_{lat}_{bs}_{chm}_{lr}_{'_'.join(var for var in vars)}_{use_noise}"+\
            f"_res_{2**(i+1)}_norma_{norma}_N_{N}_{ckpt}" + ("_no_mean" if no_mean else "")
        name += "_mean_pert" if mean_pert else ""
        np.save(rgb_folder_path + f"{name}", concat_normalised)



def load_rgbs_concat(program, lat=512, bs=16, chm=2, lr=0.002, ckpt=147000, vars=['u','v','t2m'], 
                rgb_level=3, use_noise=True, dom_size=128, res=128, N=65536, norma="classic", 
                no_mean=False, mean_pert=False, nb_samples=128,
                rgb_path="/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/Finished_experiments/"):
    
    # this function is intended to replace load_rgb everywhere
    # the goal is to compute normed rgb and delete all individual rgb to free space
    
    assert not (no_mean and mean_pert), "You can't have no_mean and mean_pert activated at the same time"
        
    data_dir = rgb_path\
                + f"stylegan2_stylegan_dom_{dom_size}_lat-dim_{lat}_bs_{bs}_{lr}_{lr}_ch-mul_{chm}"
    data_dir += f"_vars_{'_'.join(var for var in vars)}_noise_{use_noise}" + ("_no_mean" if no_mean else "")
    data_dir += "_mean_pert" if mean_pert else ""
    data_dir += "/Instance_1/toRGB_outs/"

    size_str = f"{dom_size}x{dom_size}"

    name = f"rgb_lvl_{rgb_level}_{size_str}_{lat}_{bs}_{chm}_{lr}_{'_'.join(var for var in vars)}_{use_noise}"+\
            f"_res_{res}_norma_{norma}_N_{N}_{ckpt}" + ("_no_mean" if no_mean else "")
    name += "_mean_pert" if mean_pert else ""

    if not os.path.isfile(data_dir + name):
        compute_normed_rgb(program=program, lat=lat, bs=bs, chm=chm, lr=lr, ckpt=ckpt, vars=vars, dom_size=dom_size,
                           rgb_level=rgb_level, use_noise=use_noise, norma=norma, no_mean=no_mean, mean_pert=mean_pert,
                           rgb_path=rgb_path)
    
    bigfile = np.load(data_dir+name)

    res = {}

    for key, value in program.items():
        if value[0]==1:
            
            idxList = random.sample(range(0, bigfile.shape[0]), nb_samples)
            fileList = [bigfile[i] for i in idxList]
            
            res[key]=fileList
    
    return res, data_dir



def plot_samples(lat, bs, chm, ckpt, out_dir, lr=0.002, cmap_wind='viridis', dom_size=128, no_mean=False, mean_pert=False,
            cmap_t='rainbow', vars=['u','v','t2m'], use_noise=True, nb_samples=3, cherry_picked=False, 
            exp_path="/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/Finished_experiments/",
            datapath="/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/", 
            datapath_1024=None,
            home_path="/home/poulainauzeaul/home-priam-sidev/"):
    
    from projection_artistic import artistic as art
    import numpy as np


    assert not (no_mean and mean_pert), "You can't have no_mean and mean_pert activated at the same time"
    

    if datapath_1024 is not None and dom_size in [512, 1024]:
        data_dir_0 = datapath_1024 + "IS_1_1.0_0_0_0_0_0_1024_done/"
        Means = np.load(data_dir_0 + "Mean_4_var.npy")[[var_dict[var] for var in vars]].reshape(len(vars),1,1)
        Maxs = np.load(data_dir_0 + "Max_4_var.npy")[[var_dict[var] for var in vars]].reshape(len(vars),1,1)
    
    elif dom_size in [128, 256]:
        data_dir = datapath + ("IS_1_1.0_0_0_0_0_0_256_mean_0/" if no_mean \
                                else "IS_1_1.0_0_0_0_0_0_256_mean_pert/" if mean_pert \
                                else "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/")

        data_dir_0 = datapath + "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/"
            
        mode_name = "_with_0_mean" if no_mean else "_with_8_var"
        
        means_fname = f'mean{mode_name}.npy'
        maxs_fname = f'max{mode_name}.npy'

        if not mean_pert:
            Means = np.load(data_dir + means_fname)[[var_dict[var] for var in vars]].reshape(len(vars),1,1)
            Maxs = np.load(data_dir + maxs_fname)[[var_dict[var] for var in vars]].reshape(len(vars),1,1)
        else:
            var_idx = [var_dict[var] for var in vars]
            var_idx += [vi+8 for vi in var_idx]
            Means = np.load(data_dir + means_fname)[[var_id for var_id in var_idx]].reshape(len(var_idx),1,1)
            Maxs = np.load(data_dir + maxs_fname)[[var_id for var_id in var_idx]].reshape(len(var_idx),1,1)
    else:
        print("You need to provide a path for the 1024 data")
        return 0


    Stds = (1.0/0.95) * Maxs

    real_file = random.sample(glob(data_dir_0+f"_sample*"), 1)[0] 
    print("real: ", os.path.basename(real_file))
    real_sample = np.load(real_file)[[var_dict[var] for var in vars]]
    real_sample = real_sample.reshape(-1, *real_sample.shape)

    if dom_size==128:
        real_sample = real_sample[:, :, 78:206, 55:183]
    elif dom_size==1024:
        mask = np.ma.masked_values(real_sample, 9999.0)
        real_sample = np.ma.filled(mask, fill_value=mask.mean(axis=(0,2,3)).reshape(len(vars),1,1))
        real_sample = real_sample[:, :, 153:870, :]


    fsamp_dir = exp_path +\
            f"stylegan2_stylegan_dom_{dom_size}_lat-dim_{lat}_bs_{bs}_{lr}_{lr}_ch-mul_{chm}"
    fsamp_dir += f"_vars_{'_'.join(var for var in vars)}_noise_{use_noise}"
    fsamp_dir += "_no_mean" if no_mean else ""
    fsamp_dir += "_mean_pert" if mean_pert else ""
    fsamp_dir += "/Instance_1/samples/"
    
    if cherry_picked:
        idx = [61,43,49,3]
        F_samples_list = [np.load(fsamp_dir+f"_Fsample_{ckpt}_163.npy")[idx[0]], np.load(fsamp_dir+f"_Fsample_{ckpt}_218.npy")[idx[1]], 
                np.load(fsamp_dir+f"_Fsample_{ckpt}_54.npy")[idx[2]], np.load(fsamp_dir+f"_Fsample_{ckpt}_186.npy")[idx[3]]]
        F_samples = np.array([F_samp*Stds+Means for F_samp in F_samples_list[:nb_samples]])
        print(idx)
    else:
        F_samples_list = glob(fsamp_dir+f"_Fsample_{ckpt}_*")
        file=random.sample(F_samples_list,1)[0]
        F_samples = np.load(file)
        idx = np.random.randint(low=0, high=F_samples.shape[0],size=nb_samples)
        F_samples = F_samples[idx]*Stds+Means
        if dom_size==1024:
            F_samples = F_samples[:, :, 153:870, :]
        if mean_pert:
            F_samp_tmp = np.zeros((nb_samples, len(vars), dom_size, dom_size))
            for i in range(F_samp_tmp.shape[1]):
                F_samp_tmp[:, i, :, :] = F_samples[:, i, :, :] + F_samples[:, i+len(vars), :, :]
            F_samples = F_samp_tmp
        print("indexes: ", idx,"\nfake: ", os.path.basename(file))
    
    pic_name = f"plot_{nb_samples}_" + ("rdm" if not cherry_picked else f"cherry") +f"_sample_dom_{dom_size}_lat_{lat}_bs_{bs}_"+\
                f"chm_{chm}_lr_{lr}_ckpt_{ckpt}_{'_'.join(var for var in vars)}_noise_{use_noise}"
    pic_name += "_no_mean.png" if no_mean else "_mean_pert.png" if mean_pert else ".png"
    col_titles = ["AROME-EPS"] + ["GAN" for i in range(F_samples.shape[0])]

    suptitle = f"Randomly generated samples"
    data = np.concatenate((real_sample, F_samples), axis=0)
    
    zone = "SE_for_GAN" if dom_size==128 else "SE_GAN_extend" if dom_size==256 else "AROME_all"
    can = art.canvasHolder(zone, nb_lon=dom_size, nb_lat=np.min([dom_size, 717]), fpath=home_path)

    var_names = [(var, var_dict_unit[var]) for var in vars]
    can.plot_abs_error_sev_cbar(data=data, var_names=var_names, suptitle=suptitle,
                        plot_dir=out_dir, pic_name=pic_name, col_titles=col_titles, cmap_wind=cmap_wind, cmap_t=cmap_t)



def plot_rgb(lat, bs, chm, ckpt, out_dir, lr=0.002, cmap_wind='viridis', dom_size=128, nb_res_to_plot=None, no_mean=False, 
     mean_pert=False, cmap_t='rainbow', vars=['u','v','t2m'], rgb_level=3, use_noise=True, 
     rgb_path="/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/Finished_experiments/", 
     datapath="/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/",
     home_path="/home/poulainauzeaul/home-priam-sidev/"):
    
    from projection_artistic import artistic as art
    assert not (no_mean and mean_pert), "You can't have no_mean and mean_pert activated at the same time"

    up_mode='nearest'

    if nb_res_to_plot is None:
        nb_res_to_plot = int(np.log2(dom_size))-1

    img_index = np.random.randint(0, 256) if dom_size==128 else np.random.randint(0, 128)
    program = {0: (1, 256 if dom_size==128 else 512)}

    data_dir = datapath + ("IS_1_1.0_0_0_0_0_0_256_mean_0/" if no_mean \
                             else "IS_1_1.0_0_0_0_0_0_256_mean_pert/" if mean_pert \
                             else "IS_1_1.0_0_0_0_0_0_256_done_with_8_var/")
        
    mode_name = "_with_0_mean" if no_mean else "_with_8_var"
    
    means_fname = f'mean{mode_name}.npy'
    maxs_fname = f'max{mode_name}.npy'


    if not mean_pert:
        Means = np.load(data_dir + means_fname)[[var_dict[var] for var in vars]].reshape(len(vars),1,1)
        Maxs = np.load(data_dir + maxs_fname)[[var_dict[var] for var in vars]].reshape(len(vars),1,1)
    else:
        var_idx = [var_dict[var] for var in vars]
        var_idx += [vi+8 for vi in var_idx]
        Means = np.load(data_dir + means_fname)[[var_id for var_id in var_idx]].reshape(len(var_idx),1,1)
        Maxs = np.load(data_dir + maxs_fname)[[var_id for var_id in var_idx]].reshape(len(var_idx),1,1)

    Stds = (1.0/0.95) * Maxs

    res, _ = load_rgbs(program=program, lat=lat, bs=bs, chm=chm, lr=lr, ckpt=ckpt, vars=vars, rgb_level=rgb_level, 
                        use_noise=use_noise, dom_size=dom_size, no_mean=no_mean, mean_pert=mean_pert, rgb_path=rgb_path)
    
    res_to_plot = np.load(res[0][0], allow_pickle=True).item()
    nb_res = len(res_to_plot)

    pic_name = f"plot_rdm_rgb_dom_{dom_size}_level_{rgb_level}_lat_{lat}_bs_{bs}_chm_{chm}_lr_{lr}_ckpt_{ckpt}_{up_mode}_upsamp"\
                + f"_{'_'.join(var for var in vars)}_noise_{use_noise}_{nb_res_to_plot}_res"
    pic_name += "_no_mean.png" if no_mean else "_mean_pert.png" if mean_pert else ".png"

    col_titles = [(2**(i+2),2**(i+2)) for i in range(nb_res-nb_res_to_plot, nb_res)]

    suptitle = f"Random input after each RGB layer (upsampled to {dom_size}x{dom_size})"
    
    
    data = np.zeros((nb_res_to_plot, len(vars), dom_size, dom_size))
    Upsampler = torch.nn.Upsample(size=dom_size, mode=up_mode)
    for i, key in enumerate(list(res_to_plot.keys())[-nb_res_to_plot:]):
        img = torch.from_numpy(res_to_plot[key][img_index,:,:,:])
        img = img.unsqueeze(0) if img.ndim==3 else img
        
        if not mean_pert:
            img_up = (Upsampler(img).numpy()*Stds+Means).reshape(len(vars),dom_size,dom_size)
        else:
            tmp_up = Upsampler(img).numpy()*Stds+Means
            img_up = (tmp_up[:, :len(vars), :, :] + tmp_up[:, len(vars):, :]).reshape(len(vars),dom_size,dom_size)

        data[i] = img_up#/(img_up.max(axis=(-1,-2))-img_up.min(axis=(-1,-2))).reshape(len(vars),1,1)
    

    zone = "SE_for_GAN" if dom_size==128 else "SE_GAN_extend" if dom_size==256 else "AROME_all"
    can = art.canvasHolder(zone, nb_lon=dom_size, nb_lat=np.min([dom_size, 717]), fpath=home_path)
    var_names = [(var, var_dict_unit[var]) for var in vars]
    can.plot_abs_error_sev_cbar(data=data, var_names=var_names, suptitle=suptitle,
                    plot_dir=out_dir, pic_name=pic_name, col_titles=col_titles, cmap_wind=cmap_wind, cmap_t=cmap_t)
    


def plot_pointwise_wasserstein(out_dir, lat=512, bs=16, chm=2, vars=['u','v','t2m'], no_mean=False, mean_pert=False,
                            steps=[51000,102000,147000], noise=True, dom_size=128, add_name='_16384_16384',
                            exp_path="/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/Finished_experiments/",
                            home_path="/home/poulainauzeaul/home-priam-sidev/"):    
    from projection_artistic import artistic as art

    file_loc = exp_path + f"stylegan2_stylegan_dom_{dom_size}_lat-dim_{lat}_bs_{bs}_0.002_0.002_ch-mul_{chm}_vars_" +\
                f"{'_'.join(var for var in vars)}_noise_{noise}"
    file_loc += ("_no_mean" if no_mean else "") + ("_mean_pert" if mean_pert else "") + "/Instance_1/log/"

    filenames = [file_loc + f"pw_W1_test_for_score_crawl_distance_metrics_step_{step}{add_name}.p" for step in steps]
    
    imgs = [(pickle.load(open(filename, 'rb'))[0]["pw_W1"]+pickle.load(open(filename, 'rb'))[1]["pw_W1"])/2*1e3\
                 for filename in filenames]

    var_names = [(var, "") for var in vars]
    pic_name = f"pw_wasserstein_dom_{dom_size}_vars_{'_'.join(var for var in vars)}_lat_{lat}_"+\
                f"bs_{bs}_chm_{chm}_noise_{noise}_steps_{'_'.join(str(step) for step in steps)}.png"
    
    col_titles = [f'step = {step}' for step in steps]
    suptitle = f"Pointwise Wasserstein distance (x1e3) at different steps of the training"
    zone = "SE_for_GAN" if dom_size==128 else "SE_GAN_extend" if dom_size==256 else "AROME_all"
    can = art.canvasHolder(zone, nb_lon=dom_size, nb_lat=np.min([dom_size, 717]), fpath=home_path)
    
    can.plot_abs_error(data=np.concatenate([img.reshape(-1,*img.shape) for img in imgs], axis=0), var_names=var_names,
                        plot_dir=out_dir+"pw_wasserstein_plots/", pic_name=pic_name, col_titles=col_titles, suptitle=suptitle)









## PLEIN DE TRUCS A CHANGER
"""def plot_inversion_error_comparison_lats(lat_dims, bs, chm, out_dir, date_index=3, 
                                        lead_time=3, index=0, cmap_wind='viridis', loss='mse'):
=======
##
def plot_inversion_error_comparison_lats(
    lat_dims, bs, chm, out_dir, date_index=3,
    lead_time=3, index=0, cmap_wind='viridis', loss='mse',
    mean_file='mean_with_8_var.npy',
    max_file='max_with_8_var.npy'):

>>>>>>> origin/test_merge_ddp_gan_louis:gan_horovod/plot/plotting_functions.py
    from projection_artistic import artistic as art
    data_dir = filepath + 'IS_1_1.0_0_0_0_0_0_256_done_with_8_var/'
    file_loc = filepath + 'inversion/Database_latent/W_plus/tests_lat/'
    loss = loss.lower()

    Means = np.load(data_dir + mean_file)[1:4].reshape(3,1,1)
    Maxs = np.load(data_dir + max_file)[1:4].reshape(3,1,1)        
    Stds = (1.0/0.95) * Maxs

    fake_imgs = [np.load(file_loc + f"{loss}_" + f"Fsemble_{lat_dim}_bs_{bs}_chm_{chm}_{date_index}.0"+\
                f"_{lead_time}.0.npy")[index]*Stds+Means for lat_dim in lat_dims]
    real_imgs = [np.load(file_loc + f"{loss}_" + f"Rsemble_{lat_dim}_bs_{bs}_chm_{chm}_{date_index}.0"+\
                f"_{lead_time}.0.npy")[index]*Stds+Means for lat_dim in lat_dims]

    var_names = [('u', 'm/s'), ('v', 'm/s'), ('t2m', 'K')]
    var_names = [('rr', 'mm')] + var_names if fake_imgs[0].shape[0]==4 else var_names

    with_rain = "with_rain_" if len(var_names)==4 else ''
    pic_name = f"{loss}_" + "inv_err_comp_" + with_rain + f"date_{date_index}_lt_{lead_time}_ind_{index}_lat_" +\
                     '_'.join([str(lat_dim) for lat_dim in lat_dims]) + f'_bs_{bs}_chm_{chm}.png'
    col_titles = [f'Latent dim = {lat_dim}' for lat_dim in lat_dims]
    suptitle = f"Abs error between real and inversed for differents latent dimensions\nlead time: {lead_time}, date: {date_index}," +\
                f" index: {index}, training batch size: {bs}, channel multip: {chm}," +\
                    f" loss during inv: " + f"{loss}"
    can = art.canvasHolder("SE_for_GAN", fake_imgs[0].shape[-2], fake_imgs[0].shape[-1])
    
    can.plot_abs_error(data=np.abs(np.array(fake_imgs)-np.array(real_imgs)), var_names=var_names,
                        plot_dir=out_dir+"tests_lat/", pic_name=pic_name, col_titles=col_titles, cmap_wind=cmap_wind, suptitle=suptitle)




##
def plot_inversion_error_comparison_bss(lat_dim, bss, chm, out_dir, date_index=3, 
                                        lead_time=3, index=0, cmap_wind='viridis', loss='mse',
                                        mean_file='mean_with_8_var.npy',
                                        max_file='max_with_8_var.npy'):
    from projection_artistic import artistic as art
    data_dir = filepath + 'IS_1_1.0_0_0_0_0_0_256_done_with_8_var/'
    file_loc = filepath + 'inversion/Database_latent/W_plus/tests_bs/'
    loss = loss.lower()

    Means = np.load(data_dir + mean_file)[1:4].reshape(3,1,1)
    Maxs = np.load(data_dir + max_file)[1:4].reshape(3,1,1)        
    Stds = (1.0/0.95) * Maxs

    fake_imgs = [np.load(file_loc + f"{loss}_" + f"Fsemble_{lat_dim}_bs_{bs}_chm_{chm}_{date_index}.0"+\
                f"_{lead_time}.0.npy")[index]*Stds+Means for bs in bss]
    real_imgs = [np.load(file_loc + f"{loss}_" + f"Rsemble_{lat_dim}_bs_{bs}_chm_{chm}_{date_index}.0"+\
                f"_{lead_time}.0.npy")[index]*Stds+Means for bs in bss]

    var_names = [('u', 'm/s'), ('v', 'm/s'), ('t2m', 'K')]
    var_names = [('rr', 'mm')] + var_names if fake_imgs[0].shape[0]==4 else var_names

    with_rain = "with_rain_" if len(var_names)==4 else ''
    pic_name = f"{loss}_" + "inv_err_comp_" + with_rain + f"date_{date_index}_lt_{lead_time}_ind_{index}_lat_{lat_dim}" +\
                      '_bs_' + '_'.join([str(bs) for bs in bss]) + f'_chm_{chm}.png'
    col_titles = [f'Batch size = {bs}' for bs in bss]
    suptitle = f"Abs error between real and inversed for differents batch sizes\nlead time: {lead_time}, date: {date_index}," +\
                f" index: {index}, latent dim: {lat_dim}, channel multip: {chm}," +\
                    f" loss during inv: " + f"{loss}"

    can = art.canvasHolder("SE_for_GAN", fake_imgs[0].shape[-2], fake_imgs[0].shape[-1])
    
    can.plot_abs_error(data=np.abs(np.array(fake_imgs)-np.array(real_imgs)), var_names=var_names, suptitle=suptitle,
                        plot_dir=out_dir+"tests_bs/", pic_name=pic_name, col_titles=col_titles, cmap_wind=cmap_wind)




##
def plot_inversion_error_comparison_chms(lat_dim, bs, chms, out_dir, date_index=3, 
                                        lead_time=3, index=0, cmap_wind='viridis', loss='mse',
                                        mean_file='mean_with_8_var.npy',
                                        max_file='max_with_8_var.npy'):
    from projection_artistic import artistic as art
    data_dir = filepath + 'IS_1_1.0_0_0_0_0_0_256_done_with_8_var/'
    file_loc = filepath + 'inversion/Database_latent/W_plus/tests_chm/'
    loss = loss.lower()

    Means = np.load(data_dir + mean_file)[1:4].reshape(3,1,1)
    Maxs = np.load(data_dir + max_file)[1:4].reshape(3,1,1)        
    Stds = (1.0/0.95) * Maxs

    fake_imgs = [np.load(file_loc + f"{loss}_" + f"Fsemble_{lat_dim}_bs_{bs}_chm_{chm}_{date_index}.0"+\
                f"_{lead_time}.0.npy")[index]*Stds+Means for chm in chms]
    real_imgs = [np.load(file_loc + f"{loss}_" + f"Rsemble_{lat_dim}_bs_{bs}_chm_{chm}_{date_index}.0"+\
                f"_{lead_time}.0.npy")[index]*Stds+Means for chm in chms]

    var_names = [('u', 'm/s'), ('v', 'm/s'), ('t2m', 'K')]
    var_names = [('rr', 'mm')] + var_names if fake_imgs[0].shape[0]==4 else var_names

    with_rain = "with_rain_" if len(var_names)==4 else ''
    pic_name = f"{loss}_" + "inv_err_comp_" + with_rain + f"date_{date_index}_lt_{lead_time}_ind_{index}_lat_{lat_dim}" +\
                        f'_bs_{bs}_chm_' + '_'.join([str(chm) for chm in chms]) + '.png'
    col_titles = [f'Channel multip = {chm}' for chm in chms]
    suptitle = f"Abs error between real and inversed for differents channel multipliers\nlead time: {lead_time}, date: {date_index}," +\
                f" index: {index}, training batch size: {bs}, latent dim: {lat_dim}," +\
                    f" loss during inv: " + f"{loss}"

    can = art.canvasHolder("SE_for_GAN", fake_imgs[0].shape[-2], fake_imgs[0].shape[-1])
    
    can.plot_abs_error(data=np.abs(np.array(fake_imgs)-np.array(real_imgs)), var_names=var_names, suptitle=suptitle,
                        plot_dir=out_dir+"tests_chm/", pic_name=pic_name, col_titles=col_titles, cmap_wind=cmap_wind)

##
def load_fake_real(data_dir, file_loc, lat_dim, bs, chm, date_index, 
lead_time, perturb=None, mode='lat', loss='mse',
mean_file='mean_with_8_var.npy',
max_file='max_with_8_var.npy'):

    loss = loss.lower()
    
    Means = np.load(data_dir + mean_file)[1:4].reshape(3,1,1)
    Maxs = np.load(data_dir + max_file)[1:4].reshape(3,1,1)        
    Stds = (1.0/0.95) * Maxs
    file_loc += f"tests_{mode}/"
    if perturb is not None:
        if mode=='lat':
            fake_img_perturb = np.load(file_loc + f"{loss}_" + f"Fsemble_{lat_dim}_{date_index}.0_{lead_time}.0_perturb"+\
                                f"_{float(perturb)}.npy")*Stds+Means
            real_img_pertub = np.load(file_loc + f"{loss}_" + f"Rsemble_{lat_dim}_{date_index}.0_{lead_time}.0_perturb"+\
                                f"_{float(perturb)}.npy")*Stds+Means
        else:
            fake_img_perturb = np.load(file_loc + f"{loss}_" + f"Fsemble_{lat_dim}_bs_{bs}_chm_{chm}_{date_index}.0_{lead_time}.0_perturb"+\
                                f"_{float(perturb)}.npy")*Stds+Means
            real_img_pertub = np.load(file_loc + f"{loss}_" + f"Rsemble_{lat_dim}_bs_{bs}_chm_{chm}_{date_index}.0_{lead_time}.0_perturb"+\
                                f"_{float(perturb)}.npy")*Stds+Means
    else:
        fake_img_perturb = None
        real_img_pertub = None
    if mode=='lat':
        real_img = np.load(file_loc + f"{loss}_" + f"Rsemble_{lat_dim}_{date_index}.0_{lead_time}.0.npy")*Stds+Means
        fake_img = np.load(file_loc + f"{loss}_" + f"Fsemble_{lat_dim}_{date_index}.0_{lead_time}.0.npy")*Stds+Means
    else:
        real_img = np.load(file_loc + f"{loss}_" + f"Rsemble_{lat_dim}_bs_{bs}_chm_{chm}_{date_index}.0_{lead_time}.0.npy")*Stds+Means
        fake_img = np.load(file_loc + f"{loss}_" + f"Fsemble_{lat_dim}_bs_{bs}_chm_{chm}_{date_index}.0_{lead_time}.0.npy")*Stds+Means

    return real_img, fake_img, real_img_pertub, fake_img_perturb


def plot_inversion_with_perturb(lat_dim, bs, chm, out_dir, date_index=3, lead_time=3, index=0, cmap_wind='viridis',
                                 cmap_t='rainbow', perturb = 1.1, mode='lat', loss='mse'):
    from projection_artistic import artistic as art
    data_dir = filepath + 'IS_1_1.0_0_0_0_0_0_256_done_with_8_var/'
    file_loc = filepath + 'inversion/Database_latent/W_plus/'
    loss = loss.lower()

    real_img, fake_img, real_img_pertub, fake_img_perturb = load_fake_real(data_dir, file_loc, lat_dim, bs, chm,
                                                            date_index, lead_time, perturb, mode=mode, loss=loss)
    if isinstance(index, int):
        real_img = real_img[index].reshape(-1, *real_img.shape[1:])
        fake_img = fake_img[index].reshape(-1, *fake_img.shape[1:])

        fake_img_perturb = fake_img_perturb[index].reshape(-1, *fake_img_perturb.shape[1:])
        real_img_pertub = real_img_pertub[index].reshape(-1, *real_img_pertub.shape[1:])
        msg = f'_index_{index}'
    else:
        real_img = real_img.mean(axis=0).reshape(-1, *real_img.shape[1:])
        fake_img = fake_img.mean(axis=0).reshape(-1, *fake_img.shape[1:])
        
        fake_img_perturb = fake_img_perturb.mean(axis=0).reshape(-1, *fake_img_perturb.shape[1:])
        real_img_pertub = real_img_pertub.mean(axis=0).reshape(-1, *real_img_pertub.shape[1:])

        msg = '_mean'

    data = np.concatenate((real_img, fake_img, real_img_pertub,fake_img_perturb), axis=0)    

    var_names = [('u', 'm/s'), ('v', 'm/s'), ('t2m', 'K')]
    var_names = [('rr', 'mm')] + var_names if fake_img[0].shape[1]==4 else var_names

    with_rain = "with_rain_" if len(var_names)==4 else ''
    pic_name = f"{loss}_" + "inv_err_comp_" + with_rain + f"date_{date_index}_lt_{lead_time}_lat_{lat_dim}_" +\
                f"bs_{bs}_chm_{chm}_" + msg + "_perturb_" + str(perturb) + '.png'
    
    col_titles = ["Real", "Real inversed", f"Real with \nperturbation (x{perturb})", "Perturbed real \ninversed"]
    
    title_for_ind = index if isinstance(index, int) else "mean across 16 outputs"
    
    suptitle = f"Real and inversed images with and without perturbation\nlead time: {lead_time}, date: {date_index}," +\
                f" index: {title_for_ind}, latent dim: {lat_dim}, training batch size: {bs}, channel multip: {chm}," +\
                    f" loss during inv: " + f"{loss}"

    can = art.canvasHolder("SE_for_GAN", fake_img.shape[-2], fake_img.shape[-1])
    
    can.plot_abs_error(data=data, var_names=var_names,plot_dir=out_dir+f"tests_{mode}/", suptitle=suptitle,
                    pic_name=pic_name, col_titles=col_titles, cmap_wind=cmap_wind, cmap_t=cmap_t)
    

def plot_abs_error_perturb(lat_dim, bs, chm, out_dir, date_index=3, lead_time=3, index=0, 
                            cmap_wind='Reds', perturb = 1.1, mode='lat', loss='mse'):
    from projection_artistic import artistic as art
    data_dir = filepath + 'IS_1_1.0_0_0_0_0_0_256_done_with_8_var/'
    file_loc = filepath + 'inversion/Database_latent/W_plus/'
    loss = loss.lower()
    
    real_img, fake_img, real_img_pertub, fake_img_perturb = load_fake_real(data_dir, file_loc, lat_dim, bs, chm,
                                                            date_index, lead_time, perturb, mode=mode, loss=loss)
    if isinstance(index, int):
        real_img = real_img[index].reshape(-1, *real_img.shape[1:])
        fake_img = fake_img[index].reshape(-1, *fake_img.shape[1:])

        fake_img_perturb = fake_img_perturb[index].reshape(-1, *fake_img_perturb.shape[1:])
        real_img_pertub = real_img_pertub[index].reshape(-1, *real_img_pertub.shape[1:])

        msg = f'_index_{index}'
        col_titles = [f"Abs error (index {index})\nnon perturbed", f"Abs error (index {index}) with\nperturbation (x{perturb})"]
    else:
        real_img = real_img.mean(axis=0).reshape(-1, *real_img.shape[1:])
        fake_img = fake_img.mean(axis=0).reshape(-1, *fake_img.shape[1:])
        
        fake_img_perturb = fake_img_perturb.mean(axis=0).reshape(-1, *fake_img_perturb.shape[1:])
        real_img_pertub = real_img_pertub.mean(axis=0).reshape(-1, *real_img_pertub.shape[1:])

        msg = '_mean'
        col_titles = [f"Mean abs error\nnon perturbed", f"Mean abs error with\nperturbation (x{perturb})"]

    var_names = [('u', 'm/s'), ('v', 'm/s'), ('t2m', 'K')]
    var_names = [('rr', 'mm')] + var_names if fake_img[0].shape[1]==4 else var_names
    data = np.concatenate((np.abs(real_img-fake_img), np.abs(real_img_pertub-fake_img_perturb)), axis=0)

    with_rain = "with_rain_" if len(var_names)==4 else ''
    pic_name =  f"{loss}_" + "abs_error_" + with_rain + f"date_{date_index}_lt_{lead_time}_lat_{lat_dim}_" +\
                f"bs_{bs}_chm_{chm}" + msg + "_no_perturb_v_perturb_" + str(perturb) + '.png'
    title_for_ind = index if isinstance(index, int) else "mean across 16 outputs"
    suptitle = f"Abs error comparison between real and inversed with and without perturbation\nlead time: {lead_time}, Date: {date_index}," +\
                f" index: {title_for_ind}, latent dim: {lat_dim}, training batch size: {bs}, channel multip: {chm}," +\
                    f" loss during inv: {loss}"

    can = art.canvasHolder("SE_for_GAN", fake_img.shape[-2], fake_img.shape[-1])
    
    can.plot_abs_error(data=data, var_names=var_names, plot_dir=out_dir+f"tests_{mode}/", suptitle=suptitle,
                        pic_name=pic_name, col_titles=col_titles, cmap_wind=cmap_wind)


def plot_mse(lat_dim, out_dir, label=None, bs=16, chm=2, mode='lat', date_index=3, lead_time=3, perturbs = [1.0, 1.1], 
            ax=None, fig=None, title=None, loss='mse', other_loss=False):
    
    file_loc = filepath + 'inversion/Database_latent/W_plus/'
    folder = f"tests_{mode}/"
    file_loc += folder
    loss = loss.lower()

    if ax is None or fig is None:
        fig, ax = plt.subplots(ncols=1,nrows=1, figsize=(10,10), facecolor="white")
    ax.grid(visible=True)
    if label is None:
        label = ""
    for p in perturbs:
        if p==1.0:
            label_ = label + f"perturb: none"
            loss_file = file_loc + f"{loss}_" + f"L_mse_{lat_dim}_bs_{bs}_chm_{chm}_{date_index}.0_{lead_time}.0.npy"
            if other_loss:
                mse_file = file_loc + f"mse_{loss}_" + f"L_mse_{lat_dim}_bs_{bs}_chm_{chm}_{date_index}.0_{lead_time}.0.npy"
        else:
            label_ = label + f"perturb: x{p}"
            loss_file = file_loc + f"{loss}_" + f"L_mse_{lat_dim}_bs_{bs}_chm_{chm}_{date_index}.0_{lead_time}.0_perturb_{float(p)}.npy"
            if other_loss:
                mse_file = file_loc + f"mse_{loss}_" + f"L_mse_{lat_dim}_bs_{bs}_chm_{chm}_{date_index}.0_{lead_time}.0.npy"
        to_plot = np.load(loss_file)
        ax.plot(to_plot, label=label_, marker='x' if p==1.0 else '', markersize=4, markevery=20)
        if other_loss:
            to_plot2 = np.load(mse_file)
            ax.plot(to_plot2, label=label + 'mse')
    picname = f"{loss}_" + f"inv_pert_" + "_".join(str(p) for p in perturbs) 
    picname += f"lat_{lat_dim}_bs_{bs}_chm_{chm}.png"
    ax.set_xlabel("Number of steps", size=15)
    ax.set_yscale('log')
    ax.legend(fontsize=16)
    if title is not None:
        ax.set_title(title, size=20)
        fig.savefig(out_dir+folder+picname, dpi=400)
    else:
        return fig, ax


def plot_mse_inversion_lats(lat_dims, out_dir, date_index=3, lead_time=3, perturbs = [1.0, 1.1], bs=16, chm=2, 
                            loss='mse', other_loss=False):
    
    fig, ax = plt.subplots(ncols=1,nrows=1, figsize=(10,10), facecolor="white")
    out_dir += f"tests_lat/"
    for lat_dim in lat_dims:
        label = f"lat: {lat_dim}, "
        fig, ax = plot_mse(lat_dim=lat_dim, bs=bs, chm=chm, mode='lat', out_dir=out_dir, date_index=date_index, lead_time=lead_time, 
                            label=label, perturbs = perturbs, ax=ax, fig=fig, title=None, loss=loss, other_loss=other_loss)
    ax.set_title(f"{loss} loss" + f" during the inversion\n(LT={lead_time}, DATE={date_index}, "+\
                f"LAT_DIM={lat_dims}, TRAIN BATCH={bs}, CHM={chm})", size=20)

    picname = f"{loss}_" + f"inv_pert_" + "_".join(str(p) for p in perturbs) + "_lat_" + "_".join(str(p) for p in lat_dims) +\
                 f"_bs_{bs}_chm_{chm}.png"
    fig.savefig(out_dir+picname, dpi=400, bbox_inches='tight')


def plot_mse_inversion_bss(bss, out_dir, date_index=3, lead_time=3, perturbs = [1.0, 1.1], lat_dim=512, chm=2, 
                            loss='mse', other_loss=False):
    fig, ax = plt.subplots(ncols=1,nrows=1, figsize=(10,10), facecolor="white")
    out_dir += f"tests_bs/"
    loss = loss.lower()

    for bs in bss:
        label = f"bs: {bs}, "
        fig, ax = plot_mse(bs=bs, lat_dim=lat_dim, chm=chm, mode='bs', out_dir=out_dir, date_index=date_index, lead_time=lead_time, 
                            label=label, perturbs = perturbs, ax=ax, fig=fig, title=None, loss=loss, other_loss=other_loss)
    ax.set_title(f"{loss} loss" + f" during the inversion\n(LT={lead_time}, DATE={date_index}, LAT_DIM={lat_dim}, "+\
                    f"TRAIN BATCH={bss}, CHM={chm})")

    picname = f"{loss}_" + f"inv_pert_" + "_".join(str(p) for p in perturbs) + f"_lat_{lat_dim}_bs_" +\
                 "_".join(str(p) for p in bss) + f"_chm_{chm}.png"
    fig.savefig(out_dir+picname, dpi=400, bbox_inches='tight')


def plot_mse_inversion_chms(chms, out_dir, date_index=3, lead_time=3, perturbs = [1.0, 1.1], lat_dim=512, bs=16, 
                            loss='mse', other_loss=False):
    
    fig, ax = plt.subplots(ncols=1,nrows=1, figsize=(10,10), facecolor="white")
    out_dir += f"tests_chm/"
    loss = loss.lower()

    for chm in chms:
        label = f"chm: {chm}, "
        fig, ax = plot_mse(lat_dim=lat_dim, chm=chm, bs=bs, mode='chm', out_dir=out_dir, date_index=date_index, lead_time=lead_time, 
                            label=label, perturbs = perturbs, ax=ax, fig=fig, title=None, loss=loss, other_loss=other_loss)
    ax.set_title(f"{loss} loss" + f" during the inversion\n(LT={lead_time}, DATE={date_index}, LAT_DIM={lat_dim}, "+\
                    f"TRAIN BATCH={bs}, CHM={chms})")

    picname = f"{loss}_" + f"inv_pert_" + "_".join(str(p) for p in perturbs) + f"_lat_{lat_dim}_bs_{bs}" +\
                 "_chm_" + "_".join(str(p) for p in chms) + ".png"
    fig.savefig(out_dir+picname, dpi=400, bbox_inches='tight')




<<<<<<< HEAD:gan_horovod/plotting_functions.py
=======
    res = {}

    for key, value in program.items():
        if value[0]==1:
            
            fileList=random.sample(glob_list,value[1]//256)
            
            res[key]=fileList
    
    return res, data_dir

# A CHANGER --> TOUS LES TRUCS QUI UTILISENT mean_with_orog pour leur faire accepter si on a rr 
# (ajouter une variable recensant les var est le plus cimple)
def plot_rgb(lat, bs, chm, ckpt, out_dir, lr=0.002, cmap_wind='viridis', 
            cmap_t='rainbow', vars=['u','v','t2m'], rgb_level=3, use_noise=True, up_mode='nearest',
            mean_file='mean_with_8_var.npy',
            max_file='max_with_8_var.npy'
            ):
    
    from projection_artistic import artistic as art
    dic_unit = {'u':'m/s', 'v':'m/s', 't2m':'K','rr':'mm'}
    rgb_keys = {0:'prev_rgb', 1:'prev_rgb_upsampled', 2:'input_conved', 3:'current_rgb_out'}
    rgb_key = rgb_keys[rgb_level]
    img_index = np.random.randint(0, 256)
    program = {0: (1, 256)}

    data_dir = filepath + 'IS_1_1.0_0_0_0_0_0_256_done_with_8_var/'
    Means = np.load(data_dir + mean_file)[[var_dict[var] for var in vars]].reshape(len(vars),1,1)
    Maxs = np.load(data_dir + max_file)[[var_dict[var] for var in vars]].reshape(len(vars),1,1)        

    Stds = (1.0/0.95) * Maxs

    res, path_rgb = load_rgbs(program=program, lat=lat, bs=bs, chm=chm, lr=lr, ckpt=ckpt, 
                                vars=vars, rgb_level=rgb_level, use_noise=use_noise)
    res_to_plot = np.load(res[0][0], allow_pickle=True).item()
    nb_res = len(res_to_plot)
    max_res = 2**(nb_res+1)

    pic_name = f"plot_rdm_rgb_level_{rgb_level}_lat_{lat}_bs_{bs}_chm_{chm}_lr_{lr}_ckpt_{ckpt}_{up_mode}_upsamp"\
                + f"_{'_'.join(var for var in vars)}_noise_{use_noise}.png"
    col_titles = [(2**(i+2),2**(i+2)) for i in range(nb_res)]

    suptitle = f"Random input after each RGB layer ({rgb_key} upsampled to {max_res}x{max_res})"+\
                f"\nLat dim={lat}, training batch={bs}, channel_multip={chm}, training steps={ckpt}, lr={lr}."
    
    max_res = res_to_plot[list(res_to_plot.keys()[-1])].shape[-1]
    data = np.zeros((nb_res, len(vars), max_res, max_res))
    for i in res_to_plot.keys():
        img = torch.from_numpy(res_to_plot[i][img_index,:,:,:])
        img = img.unsqueeze(0) if img.ndim==3 else img
        Upsampler = torch.nn.Upsample(size=max_res, mode=up_mode)
        img_up = (Upsampler(img).numpy()*Stds+Means).reshape(len(vars),max_res,max_res)
        data[i-1] = img_up#/(img_up.max(axis=(-1,-2))-img_up.min(axis=(-1,-2))).reshape(len(vars),1,1)
    

    can = art.canvasHolder("SE_for_GAN", data.shape[-2], data.shape[-1])
    var_names = [(var, dic_unit[var]) for var in vars]
    can.plot_abs_error_sev_cbar(data=data, var_names=var_names, suptitle=suptitle,
                        plot_dir=out_dir, pic_name=pic_name, col_titles=col_titles, cmap_wind=cmap_wind, cmap_t=cmap_t)
>>>>>>> origin/test_merge_ddp_gan_louis:gan_horovod/plot/plotting_functions.py


def plot_mean_map(program, out_dir, lat=512, bs=16, chm=2, lr=0.002, ckpt=147000, vars=['u','v','t2m'], 
                    upsampling_mode='nearest', rgb_level=3,
                    mean_file='mean_with_8_var.npy',
                    max_file='max_with_8_var.npy'):
    from projection_artistic import artistic as art
    dic_unit = {'u':'m/s', 'v':'m/s', 't2m':'K','rr':'mm'}
    rgb_keys = {0:'prev_rgb', 1:'prev_rgb_upsampled', 2:'input_conved', 3:'current_rgb_out'}
    rgb_key = rgb_keys[rgb_level]

    data_dir = filepath + 'IS_1_1.0_0_0_0_0_0_256_done_with_8_var/'
    Means = np.load(data_dir + mean_file)[[var_dict[var] for var in vars]].reshape(len(vars),1,1)
    Maxs = np.load(data_dir + max_file)[[var_dict[var] for var in vars]].reshape(len(vars),1,1)        

    Stds = (1.0/0.95) * Maxs

    res, path_rgb = load_rgbs(program=program, lat=lat, bs=bs, chm=chm, lr=lr, ckpt=ckpt, vars=vars, rgb_level=rgb_level)
    N = program[0][1]
    biglist = [np.load(res[0][i], allow_pickle=True).item() for i in range(len(res[0]))]
    print("data loaded")

    nb_res = len(biglist[0])
    nb_var = len(vars)
    max_res = biglist[0][nb_res].shape[-1]
    data = np.zeros((nb_res,nb_var,max_res,max_res))
    for i in range(1, nb_res+1):
        upsampler = torch.nn.Upsample(size=max_res, mode=upsampling_mode)
        concat = np.concatenate([upsampler(torch.from_numpy(biglist[j][i])).numpy() for j in range(len(biglist))])
        data[i-1] = np.mean(concat*Stds+Means, axis=0)#.reshape(-1, nb_var, 128, 128)
    
    pic_name = f"plot_mean_rgb_level_{rgb_level}_lat_{lat}_bs_{bs}_chm_{chm}_lr_{lr}_ckpt_{ckpt}_{N}_samples"\
                + f"_{'_'.join(var for var in vars)}.png"
    col_titles = [(2**(i+2),2**(i+2)) for i in range(nb_res)]

    suptitle = f"Mean image after each RGB layer ({rgb_key} upsampled to {max_res}x{max_res}) for {N} random entries to the network"+\
                f"\nLat dim={lat}, training batch={bs}, channel_multip={chm}, training steps={ckpt}, lr={lr}."

    can = art.canvasHolder("SE_for_GAN", max_res, max_res)
    var_names = [(var, dic_unit[var]) for var in vars]
    can.plot_abs_error(data=data, var_names=var_names, suptitle=suptitle,
                        plot_dir=out_dir, pic_name=pic_name, col_titles=col_titles, cmap_wind="viridis", cmap_t="rainbow")



def plot_std_map(program, out_dir, lat=512, bs=16, chm=2, lr=0.002, ckpt=147000, vars=['u','v','t2m'], 
                    upsampling_mode='nearest', rgb_level=3, use_noise=True,
                    mean_file='mean_with_8_var.npy',
                    max_file='max_with_8_var.npy'):
    from projection_artistic import artistic as art
    dic_unit = {'u':'m/s', 'v':'m/s', 't2m':'K','rr':'mm'}
    rgb_keys = {0:'prev_rgb', 1:'prev_rgb_upsampled', 2:'input_conved', 3:'current_rgb_out'}
    rgb_key = rgb_keys[rgb_level]

    data_dir = filepath + 'IS_1_1.0_0_0_0_0_0_256_done_with_8_var/'
    Means = np.load(data_dir + mean_file)[[var_dict[var] for var in vars]].reshape(len(vars),1,1)
    Maxs = np.load(data_dir + max_file)[[var_dict[var] for var in vars]].reshape(len(vars),1,1)        

    Stds = (1.0/0.95) * Maxs

    res, rgb_path = load_rgbs(program=program, lat=lat, bs=bs, chm=chm, lr=lr, ckpt=ckpt, 
                                vars=vars, rgb_level=rgb_level, use_noise=use_noise)
    N = program[0][1]
    biglist = [np.load(res[0][i], allow_pickle=True).item() for i in range(len(res[0]))]
    print("data loaded")

    nb_res = len(biglist[0])
    nb_var = len(vars)
    max_res = biglist[0][nb_res].shape[-1]
    data = np.zeros((nb_res,nb_var,max_res,max_res))
    for i in range(1, nb_res+1):
        upsampler = torch.nn.Upsample(size=max_res, mode=upsampling_mode)
        concat = np.concatenate([upsampler(torch.from_numpy(biglist[j][i])).numpy() for j in range(len(biglist))])*Stds+Means
        std_all_but_var = np.std(concat, axis=(0, 2, 3)).reshape(1,nb_var,1,1)
        concat_normalised = concat/std_all_but_var
        np.save(rgb_path + f"rgb_level_{rgb_level}_concat_res_{2**(i+1)}_normalised_by_std", concat_normalised)
        data[i-1] = np.std(concat_normalised, axis=0)
        #print(data[i-1].max(axis=(-1,-2)))
    
    pic_name = f"plot_std_rgb_level_{rgb_level}_lat_{lat}_bs_{bs}_chm_{chm}_lr_{lr}_ckpt_{ckpt}_{N}_samples"\
                + f"_{'_'.join(var for var in vars)}.png"
    col_titles = [(2**(i+2),2**(i+2)) for i in range(nb_res)]

    suptitle = f"Std renorm after each RGB layer ({rgb_key} upsampled to {max_res}x{max_res}) for {N} random entries to the network"+\
                f"\nLat dim={lat}, training batch={bs}, channel_multip={chm}, training steps={ckpt}, lr={lr}."

    can = art.canvasHolder("SE_for_GAN", max_res, max_res)
    var_names = [(var, dic_unit[var]) for var in vars]
    can.plot_abs_error(data=data, var_names=var_names, suptitle=suptitle,
                        plot_dir=out_dir, pic_name=pic_name, col_titles=col_titles, cmap_wind="viridis", cmap_t="rainbow")



def plot_mean_rand_pixel(program, out_dir, lat=512, bs=16, chm=2, lr=0.002, ckpt=147000, vars=['u','v','t2m'], 
                    upsampling_mode='nearest', rgb_level=3, pixs=None,
                    mean_file='mean_with_8_var.npy',
                    max_file='max_with_8_var.npy'):
    from projection_artistic import artistic as art
    dic_unit = {'u':'m/s', 'v':'m/s', 't2m':'K','rr':'mm'}
    rgb_keys = {0:'prev_rgb', 1:'prev_rgb_upsampled', 2:'input_conved', 3:'current_rgb_out'}
    rgb_key = rgb_keys[rgb_level]

    data_dir = filepath + 'IS_1_1.0_0_0_0_0_0_256_done_with_8_var/'
    Means = np.load(data_dir + mean_file)[[var_dict[var] for var in vars]].reshape(len(vars),1,1)
    Maxs = np.load(data_dir + max_file)[[var_dict[var] for var in vars]].reshape(len(vars),1,1)        

    Stds = (1.0/0.95) * Maxs

    res = load_rgbs(program=program, lat=lat, bs=bs, chm=chm, lr=lr, ckpt=ckpt, vars=vars, rgb_level=rgb_level)
    N = program[0][1]
    biglist = [np.load(res[0][i], allow_pickle=True).item() for i in range(len(res[0]))]
    print("data loaded")
    
    nb_res = len(biglist[0])
    max_res = 2**(nb_res+1)
    nb_var = len(vars)
    if pixs is None:
        chosen_pixels = np.random.randint(low=0, high=max_res, size=2)
        pix1, pix2 = chosen_pixels
    else:
        chosen_pixels=pixs
        pix1, pix2 = pixs
    data = np.zeros((nb_res,nb_var,max_res,max_res))
    for i in range(1, nb_res+1):
        upsampler = torch.nn.Upsample(size=(max_res,max_res), mode=upsampling_mode)
        concat = np.concatenate([upsampler(torch.from_numpy(biglist[j][i])).numpy() for j in range(len(biglist))])
        data[i-1] = np.mean(concat*Stds+Means, axis=0)#.reshape(-1, nb_var, 128, 128)
    
    data = data[:, :, pix1, pix2]
    
    pic_name = f"plot_mean_rgb_level_{rgb_level}_rand_{pix1}_{pix2}_lat_{lat}_bs_{bs}_chm_{chm}_lr_{lr}_ckpt_{ckpt}_{N}"\
                + f"_samples_{'_'.join(var for var in vars)}.png"

    suptitle = f"Mean value at a random location ({rgb_key}\nupsampled to {max_res}x{max_res}) for {N} "\
                +"random entries"+\
                f"\n(Lat dim={lat},training batch={bs}, channel_multip={chm},\ntraining steps={ckpt}, lr={lr}, location: {chosen_pixels})"

    fig, axs = plt.subplots(nrows=nb_var, ncols=1, figsize=(8, 8*nb_var), facecolor="white")
    var_names = [(var, dic_unit[var]) for var in vars]
    i = 0
    for var, unit in var_names:
        ax = axs[i] if nb_var>1 else axs
        ax.set_title(f"{var} ({unit})", fontsize=32)
        ax.set_xticks(ticks=np.arange(nb_res), labels=[2**(j+2) for j in range(nb_res)])
        ax.set_xlabel("resolution", fontsize=20)
        ax.set_ylabel("Mean value", fontsize=32)
        ax.plot(data[:, i])
        i +=1
    
    st = fig.suptitle(suptitle, fontsize=32)
    st.set_y(0.98)
    fig.subplots_adjust(bottom=0.005, top=0.88-0.07*(4-len(vars)), left=0.05, right=0.95, wspace=0.01, hspace=0.2)
    fig.savefig(out_dir+pic_name, dpi=400, bbox_inches='tight')"""








########## UNUSED ##########
"""
def plot_inversion_error(fake_img, real_img, cmap='rainbow', index=0):
    # img of shape (batch_size, channels, width, height)
    # index: used to select one img of the batch (should be between 0 and 15)
    # de-normalize imgs before entering the fucntion

    names = ['u (m/s)', 'v (m/s)', 't2m (K)']#, 'precip']
    names = ['precip (???)'] + names if fake_img.shape[0]==4 else names
    fig, axs = plt.subplots(nrows=len(names), ncols=1, figsize=(6*len(names), 12), facecolor='white', layout='constrained')
    
    for i in range(len(names)):
        ax = axs[i]
        diff = np.abs(fake_img[index]-real_img[index])[i]
        im = ax.imshow(diff, cmap=cmap, origin='lower')
        ax.set_title(names[i], fontsize=14)
        fig.colorbar(mappable=im, ax=ax)
    #fig.suptitle("Absolute error between real and fake", fontsize=18)
    #plt.tight_layout()
    #plt.subplots_adjust(top=0.95)


def plot_one_day_err(lat_dim, out_dir, date_index, indexes=[0], with_rain=False, cmap='rainbow'):
    from projection_artistic import artistic as art
    file_loc = filepath + 'inversion/Database_latent/W_plus/'
    data_dir = filepath + 'IS_1_1.0_0_0_0_0_0_256_done_with_8_var/'

    mean_day_error = np.zeros((16,4,128,128)) if with_rain else np.zeros((16,3,128,128))
    for lead_time in range(8):
        real_img, fake_img, _, _ = load_fake_real(data_dir, file_loc, lat_dim, date_index, lead_time, perturb=None)
        mean_day_error += abs(real_img-fake_img)
    mean_day_error /= 8

    var_names = [('u', 'm/s'), ('v', 'm/s'), ('t2m', 'K')]
    var_names = [('rr', '(???)')] + var_names if with_rain else var_names
    
    col_titles = [f"Mettre le jour avec pd" for i in range(len(indexes))]

    pic_name = "abs_error_across_day" + ("_with_rain_" if with_rain else '_') + \
                f"date_{date_index}_days_" + '_'.join([str(ind) for ind in indexes]) + "_lat_{lat_dim}_"'.png'
    
    data = mean_day_error[indexes, :, :, :]
    can = art.canvasHolder("SE_for_GAN", 128, 128)
    
    can.plot_abs_error(data=data, var_names=var_names,
                        plot_dir=out_dir, pic_name=pic_name, col_titles=col_titles, cmap=cmap)
"""
