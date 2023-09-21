import metrics_plotting as mp
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
metrics_path = '/scratch/mrmn/brochetc/GAN_2D/Exp_StyleGAN_final/Set_1/stylegan2_stylegan_dom_256_lat-dim_512_bs_4_0.002_0.002_ch-mul_2_vars_u_v_t2m_noise_True/Instance_14/log/'
real_data_dir = '/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_large_lt_done/'
dic_1 = pickle.load(open(metrics_path + 'final_exp_distance_metrics_step_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_16384.p', 'rb'))
print(dic_1.keys())
output_dir = metrics_path

 #for step in list(range(25)):
 #   mp.wasserstein_maps(dic_1, output_dir, step=step)


w1 = np.zeros((25,))
delta_w1 = np.zeros((25,))
for step in list(range(25)):
    data = np.array(dic_1[step]['W1_Center_NUMPY'])
    w1[step] = data.mean(axis=0)
    delta_w1[step] = data.std(axis=0)
print(w1[-1]) 

plt.plot(range(25),w1, 'bo-', linewidth = 2.5)
plt.fill_between(np.arange(25), w1-delta_w1, w1 + delta_w1, alpha=0.2, color = 'blue')
plt.title('W1 Center')
plt.xlabel('epoch')
plt.savefig(f'{metrics_path}W1_evolution.png')
plt.close()

SWD = np.zeros((25,5))
delta_SWD = np.zeros((25,5))
for step in list(range(25)):
    data = np.array([np.array(arr) for arr in dic_1[step]['SWD_metric_torch']])
    SWD[step] = data.mean(axis=0)
    delta_SWD[step] = data.std(axis=0) 
print(SWD[-1])
color = ['bo-','go-', 'ro-', 'ko-', 'ko-']

for scale in range(4):
    plt.plot(range(25),SWD[:,scale], color[scale], linewidth = 2.5)
    plt.fill_between(np.arange(25),SWD[:,scale]-delta_SWD[:,scale], SWD[:,scale] + delta_SWD[:,scale], alpha=0.2, color = 'blue')
    plt.title(f'SWD coarsened x{scale}')
    plt.xlabel('epoch')
    plt.savefig(f'{metrics_path}SWD_coarsened_x{scale}_evolution.png')
    plt.close()

scale= 4

plt.plot(range(25),SWD[:,scale], color[scale], linewidth = 2.5)
plt.fill_between(np.arange(25),SWD[:,scale]-delta_SWD[:,scale], SWD[:,scale] + delta_SWD[:,scale], alpha=0.2, color = 'blue')
plt.title(f'SWD coarsened x{scale}')
plt.xlabel('epoch')
plt.savefig(f'{metrics_path}SWD_avg_evolution.png')
plt.close()

"""for step in list(range(25)):
    mp.multivariate_correlation(dic_1, output_dir, step=step)"""
var  =['u','v','t2m']
dic_1 = pickle.load(open(metrics_path + 'final_exp_standalone_metrics_step_0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_16384.p', 'rb'))
real_corr = pickle.load(open(f'{real_data_dir}dom_256_standalone_metrics_u_v_t2m_dom_256_16384.p', 'rb'))['ls_metric']
mae_corr = np.zeros((25,3))
for step in list(range(25)):
    print(step)
    mae_corr[step] = mp.length_scales_maps(dic_1, real_corr, output_dir, step=step)
for i in range(3):
    plt.plot(range(25),mae_corr[:,i], 'bo-', linewidth = 2.5)
    #plt.fill_between(np.arange(25),SWD[:,scale]-delta_SWD[:,scale], SWD[:,scale] + delta_SWD[:,scale], alpha=0.2, color = 'blue')
    plt.title(f'Corr length MAE')
    plt.xlabel('epoch')
    plt.savefig(f'{metrics_path}mae_ls_{var[i]}.png')
    plt.close()

real_spectrum = pickle.load(open(f'{real_data_dir}dom_256_standalone_metrics_u_v_t2m_dom_256_16384.p', 'rb'))['spectral_compute']
PSD = np.zeros((25,3))
for step in list(range(25)):
    print(step)
    PSD[step] = mp.spectral_plot(dic_1, real_spectrum, output_dir, step=step)
for i in range(3):
    plt.plot(range(25),PSD[:,i], 'bo-', linewidth = 2.5)
    #plt.fill_between(np.arange(25),SWD[:,scale]-delta_SWD[:,scale], SWD[:,scale] + delta_SWD[:,scale], alpha=0.2, color = 'blue')
    plt.title(f'PSD error (dB)')
    plt.xlabel('epoch')
    plt.savefig(f'{metrics_path}psd_err_{var[i]}.png')
    plt.close()
print(PSD[-1])