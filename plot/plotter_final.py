import plot.plotting_functions as plf
from time import perf_counter
from multiprocessing import Pool


path = "/scratch/mrmn/brochetc/Exp_StyleGAN_final/Set_1/"
multip = False # set to true if you want to do multiprocessing 
              # (test several times on x samples and compute mean + std for several lat_dim)
test_oscil = True

data_dir = "/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/"
metrics_names = ["multivar", "pw_W1", "W1_random", "W1_Center", "SWD_metric_torch",
                "ls_dist_torch", "spectral_dist_torch"]
num_samples = 10
nb_ckpt = 25
t = perf_counter()
if not multip:
    i = 4
    latent_dim = 512
    output_dir = path + f"stylegan2_stylegan_lat-dim_{lat_dim}_bs_{i}_0.002_0.002_ch-mul_2_vars_u_v_t2m_noise_True/Instance_14"
    plf.plot_metrics_after_training(data_dir=data_dir, output_dir=output_dir, latent_dim=latent_dim, 
                            num_samples=num_samples, metrics_names=metrics_names, nb_ckpt=nb_ckpt, test_oscil=test_oscil)
    
    t_f = perf_counter()
    msg = f"Time taken by the test ({num_samples} samples/{nb_ckpt} ckpt): {t_f-t}s."
    if test_oscil:
        msg = f"Time taken by the test ({(16384//num_samples)*num_samples} samples seen/1 cpkt): {perf_counter()-t}s."
    print(msg)
else:
    def multip_plotter(lat_dim):
        output_dir = path + f"stylegan2_stylegan_lat-dim_{lat_dim}_bs_16_0.002_0.002_ch-mul_2_vars_u_v_t2m_noise_True/Instance_14"
        plf.plot_metrics_after_training(data_dir=data_dir, output_dir=output_dir, latent_dim=lat_dim, 
                            num_samples=num_samples, metrics_names=metrics_names, nb_ckpt=nb_ckpt, test_oscil=True)
    
    lat_dims = [64,128,256,1024]
    with Pool(min(8, len(lat_dims))) as p:
        res = p.map(multip_plotter, lat_dims)
    
    print(f"Time taken by the test ({num_samples} samples/{nb_ckpt} ckpt): {perf_counter()-t}s.")