import torch
import numpy as np
from glob import glob
import os
import pickle
from time import perf_counter
import random

filepath = "/home/poulainauzeaul/scratch-priam-sidev/Exp_StyleGAN/"

################ reference dictionary to know what variables to sample where
################ do not modify unless you know what you are doing 
var_dict={'rr' : 0, 'u' : 1, 'v' : 2, 't2m' :3 , 'orog' : 4, 'z500': 5, 't850': 6, 'tpw850': 7}
var_dict_unit={'rr': "mm", 'u': "m/s", 'v': "m/s", 't2m': "K" , 'orog': "m", 'z500': "m", 't850': "K", 'tpw850': "K"}


def samples_without_noise_influence(lat=512, bs=16, chm=2, lr=0.002, ckpt=147000, vars=['u','v','t2m'], dom_size=128,
                    use_noise=True, fpath=None, noise_max=1e-4):

    device = torch.device('cuda')
    if fpath is not None:
        Filepath = fpath
    else:
        Filepath = filepath
    
    if not os.path.isfile(Filepath + "rgb_no_noise/" + f"dom_{dom_size}_lat_{lat}_bs_{bs}_chm_{chm}_"+\
        f"vars_{'_'.join(v for v in vars)}_noise_{use_noise}_max_{noise_max}_ckpt_{ckpt}"):

        import stylegan2 as RN

        output_dir = Filepath + "Finished_experiments/"+\
            f"stylegan2_stylegan_dom_{dom_size}_lat-dim_{lat}_bs_{bs}_{lr}_{lr}_ch-mul_{chm}"
        output_dir += f"_vars_{'_'.join(str(var_name) for var_name in vars)}"
        output_dir += f"_noise_{use_noise}/Instance_1"

        ckpt_path = output_dir + '/models/'
        ckpt_name = str(ckpt).zfill(6) + '.pt'
        
        nb_weights = int(np.log2(dom_size))*2-4

        ckpt_dic = torch.load(ckpt_path+ckpt_name, map_location=device)
        
        model_names =  RN.library['stylegan2']
        modelG_n = getattr(RN, model_names['G'])
        modelG_ema1 = modelG_n(size=dom_size, style_dim=lat, n_mlp=8, channel_multiplier=chm, nb_var=len(vars), use_noise=use_noise)
        modelG_ema1 = modelG_ema1.to(device)

        modelG_ema2 = modelG_n(size=dom_size, style_dim=lat, n_mlp=8, channel_multiplier=chm, nb_var=len(vars), use_noise=use_noise)
        modelG_ema2 = modelG_ema1.to(device)


        modelG_ema1.load_state_dict(ckpt_dic["g_ema"])
        modelG_ema1.eval()

        # cap the noise to some value and charge it into another network
        for j in range(nb_weights):
            w_current = ckpt_dic['g_ema'][f"convs.{j}.noise.weight"].cpu()
            if w_current.abs()<torch.tensor([noise_max]):
                w_new = w_current.clone()
            else:
                w_new = torch.tensor([noise_max])
            ckpt_dic['g_ema'][f"convs.{j}.noise.weight"] = w_new
            print("before: ", w_current.item(), "\nafter: ", w_new.item(), '\n')

        modelG_ema2.load_state_dict(ckpt_dic["g_ema"])
        modelG_ema2.eval()
        
        nb_batch = 128 if not dom_size==128 else 256
        samp_size = 128 if dom_size==128 else 64
        ts = perf_counter()
        for i in range(nb_batch):
            samples_dic = {}
            samples_dic["normal"] = []
            samples_dic["noise_capped"] = []
            if (i+1)%16==0:
                print(f"Batch {i+1}/{nb_batch}: {perf_counter()-ts}s")
                ts = perf_counter()
            z = torch.empty(samp_size, lat).normal_().to(device)
            rgb_normal = modelG_ema1([z])[0].detach().cpu().numpy()
            samples_dic["normal"].append(rgb_normal)
            
            rgb_noise_capped = modelG_ema2([z])[0].detach().cpu().numpy()
            samples_dic["noise_capped"].append(rgb_noise_capped)

            for key in list(samples_dic.keys()):
                concat = np.concatenate(samples_dic[key], axis=0)
                samples_dic[key] = concat
                
            filename = Filepath + "rgb_no_noise/" + f"noise_capped_dom_{dom_size}_lat_{lat}_bs_{bs}_chm_{chm}_vars_{'_'.join(v for v in vars)}"+\
                        f"_noise_{use_noise}_max_{noise_max}_ckpt_{ckpt}_{i}.p"
            pickle.dump(samples_dic, open(filename, 'wb'))



def modify_noise_per_scale(lat=512, bs=16, chm=2, lr=0.002, ckpt=147000, vars=['u','v','t2m'], dom_size=128,
                    use_noise=True, fpath=None, scale=[1], new_weights=[1e-4]):

    # scale: list of distinct ints from 1 to (log2(dom_size)-2)*2-1
    scale = [s for s in np.sort(np.unique(scale)) if s>0 and s<=np.log2(dom_size)-2]
    if len(new_weights)!=len(scale):
        new_weights = [new_weights[0] for i in range(len(scale))]
    device = torch.device('cuda')
    if fpath is not None:
        Filepath = fpath
    else:
        Filepath = filepath
    
    import stylegan2 as RN

    output_dir = Filepath + "Finished_experiments/"+\
        f"stylegan2_stylegan_dom_{dom_size}_lat-dim_{lat}_bs_{bs}_{lr}_{lr}_ch-mul_{chm}"
    output_dir += f"_vars_{'_'.join(str(var_name) for var_name in vars)}"
    output_dir += f"_noise_{use_noise}/Instance_1"

    ckpt_path = output_dir + '/models/'
    ckpt_name = str(ckpt).zfill(6) + '.pt'
    
    ckpt_dic = torch.load(ckpt_path+ckpt_name, map_location=device)
    ckpt_dic_mod = torch.load(ckpt_path+ckpt_name, map_location=device)
    
    model_names =  RN.library['stylegan2']
    modelG_n = getattr(RN, model_names['G'])
    modelG_ema1 = modelG_n(size=dom_size, style_dim=lat, n_mlp=8, channel_multiplier=chm, nb_var=len(vars), use_noise=use_noise)
    modelG_ema1 = modelG_ema1.to(device)

    modelG_ema2 = modelG_n(size=dom_size, style_dim=lat, n_mlp=8, channel_multiplier=chm, nb_var=len(vars), use_noise=use_noise)
    modelG_ema2 = modelG_ema1.to(device)


    modelG_ema1.load_state_dict(ckpt_dic["g_ema"])
    modelG_ema1.eval()

    # cap the noise to some value and charge it into another network
    for i, s in enumerate(scale):
        print("Scale: ", s)
        w_current = ckpt_dic_mod['g_ema'][f"convs.{2*(s-1)}.noise.weight"].to(device)
        w_new = torch.tensor([new_weights[i]]).to(device)
        ckpt_dic_mod['g_ema'][f"convs.{2*(s-1)}.noise.weight"] = w_new
        print("before: ", w_current.item(), "\nafter: ", w_new.item())

        w_current = ckpt_dic['g_ema'][f"convs.{2*(s-1)+1}.noise.weight"].cpu()
        w_new = torch.tensor([new_weights[i]])
        ckpt_dic['g_ema'][f"convs.{2*(s-1)+1}.noise.weight"] = w_new
        print("before: ", w_current.item(), "\nafter: ", w_new.item(), '\n')

    modelG_ema2.load_state_dict(ckpt_dic_mod["g_ema"])
    modelG_ema2.eval()
    
    nb_batch = 16 # change that to generate more samples
    samp_size = 64
    ts = perf_counter()
    for i in range(nb_batch):
        samples_dic = {}
        samples_dic["normal"] = []
        samples_dic["noise_changed"] = []
        if (i+1)%16==0:
            print(f"Batch {i+1}/{nb_batch}: {perf_counter()-ts}s")
            ts = perf_counter()
        with torch.no_grad():
            z = torch.empty(samp_size, lat).normal_().to(device)
            rgb_normal = modelG_ema1([z])[0].cpu().numpy()
            samples_dic["normal"].append(rgb_normal)
            
            rgb_noise_changed = modelG_ema2([z])[0].cpu().numpy()
            samples_dic["noise_changed"].append(rgb_noise_changed)

        for key in list(samples_dic.keys()):
            concat = np.concatenate(samples_dic[key], axis=0)
            samples_dic[key] = concat
            
        filename = Filepath + "rgb_no_noise/" + f"noise_scales_{'_'.join(str(s) for s in scale)}_"+\
            f"{'_'.join(str(w) for w in new_weights)}_dom_{dom_size}_lat_{lat}_bs_{bs}_chm"+\
            f"_{chm}_vars_{'_'.join(v for v in vars)}_noise_{use_noise}_ckpt_{ckpt}_{i}.p"
        pickle.dump(samples_dic, open(filename, 'wb'))



def plot_noise_change_scale(lat=512, bs=16, chm=2, lr=0.002, ckpt=147000, vars=['u','v','t2m'], dom_size=128,
                    use_noise=True, fpath=None, scale=[1], new_weights=[1e-4], reproduce=False):

    from projection_artistic import artistic as art

    # scale: list of distinct ints from 1 to (log2(dom_size)-2)*2-1
    scale = [s for s in np.sort(np.unique(scale)) if s>0 and s<=np.log2(dom_size)-2]
    if len(new_weights)!=len(scale):
        new_weights = [new_weights[0] for i in range(len(scale))]

    if fpath is not None:
        Filepath = fpath
    else:
        Filepath = filepath

    rgb_dir = Filepath + "rgb_no_noise/"
    data_dir = Filepath + 'IS_1_1.0_0_0_0_0_0_256_done_with_8_var/'
    
    Means = np.load(data_dir + 'mean_with_8_var.npy')[[var_dict[var] for var in vars]].reshape(len(vars),1,1)
    Maxs = np.load(data_dir + 'max_with_8_var.npy')[[var_dict[var] for var in vars]].reshape(len(vars),1,1)        

    Stds = (1.0/0.95) * Maxs

    rgb_fname = f"noise_scales_{'_'.join(str(s) for s in scale)}_"+\
                f"{'_'.join(str(w) for w in new_weights)}_dom_{dom_size}_lat_{lat}_bs_{bs}_chm"+\
                f"_{chm}_vars_{'_'.join(v for v in vars)}_noise_{use_noise}_ckpt_{ckpt}_"
    
    rgb_list = np.sort(glob(rgb_dir+rgb_fname+"*"))
    if not reproduce:
        rgb_file = random.sample(rgb_list, 1)[0]
    else:
        rgb_file = rgb_list[0]
    print("batch chosen: ", rgb_file[len(rgb_dir+rgb_fname):-2])
    
    rgb_dic = np.load(rgb_file, allow_pickle=True)
    
    if not reproduce:
        img_index = random.sample(range(0, rgb_dic["normal"].shape[0]), 2)
    else:
        img_index = [0,1]
    print("img_indexes: ", img_index)

    rgb_normal = rgb_dic["normal"][img_index]*Stds+Means
    rgb_noise_capped = rgb_dic["noise_changed"][img_index]*Stds+Means
    reprod = "reprod" if reproduce else ""

    col_titles = ["Normal noise", "With modif", "Abs difference"]*2
    suptitle = f"Random samples comparison with and without changed noises\nScale(s): {','.join(str(s) for s in scale)}, "+\
                f"new weight(s): {','.join(str(w) for w in new_weights)}"
    pic_name = f"img_noise_scales_{'_'.join(str(s) for s in scale)}_"+\
               f"{'_'.join(str(w) for w in new_weights)}__{reprod}_dom_{dom_size}_lat_{lat}_bs_{bs}_chm_{chm}_vars"+\
               f"_{'_'.join(v for v in vars)}_noise_{use_noise}_ckpt_{ckpt}.png"

    data = []
    for i in range(rgb_normal.shape[0]):
        data.append(rgb_normal[i])
        data.append(rgb_noise_capped[i])
        data.append(np.abs(rgb_normal[i]-rgb_noise_capped[i]))
        #print("Number of pixels that are different: ", np.count_nonzero(rgb_noise_capped[i]!=rgb_normal[i]))
        #print("max abs diff: ", np.abs(rgb_normal[i]-rgb_noise_capped[i]).max(axis=(-2,-1)),
        #      "\nvalue extent of non pert sample: ", rgb_normal[i].max(axis=(-2,-1))-rgb_normal[i].min(axis=(-2,-1)))
    data = np.array(data)

    can = art.canvasHolder("SE_for_GAN" if dom_size==128 else "SE_GAN_extend", dom_size, dom_size)
    var_names = [(var, var_dict_unit[var]) for var in vars]
    can.plot_abs_error_sev_cbar(data=data, var_names=var_names, suptitle=suptitle,
                                plot_dir=Filepath + "rgb_no_noise/", pic_name=pic_name, 
                                col_titles=col_titles, cmap_wind="viridis", cmap_t="rainbow")




def plot_rgb_noise_v_no_noise(lat=512, bs=16, chm=2, lr=0.002, ckpt=147000, vars=['u','v','t2m'], dom_size=128,
                    use_noise=True, fpath=None, noise_max=1e-4, reproduce=False):

    from projection_artistic import artistic as art

    if fpath is not None:
        Filepath = fpath
    else:
        Filepath = filepath

    rgb_dir = Filepath + "rgb_no_noise/"
    data_dir = Filepath + 'IS_1_1.0_0_0_0_0_0_256_done_with_8_var/'
    
    Means = np.load(data_dir + 'mean_with_8_var.npy')[[var_dict[var] for var in vars]].reshape(len(vars),1,1)
    Maxs = np.load(data_dir + 'max_with_8_var.npy')[[var_dict[var] for var in vars]].reshape(len(vars),1,1)        

    Stds = (1.0/0.95) * Maxs

    rgb_fname = f"noise_capped_dom_{dom_size}_lat_{lat}_bs_{bs}_chm_{chm}_vars_{'_'.join(v for v in vars)}"+\
                f"_noise_{use_noise}_max_{noise_max}_ckpt_{ckpt}_"
    
    rgb_list = np.sort(glob(rgb_dir+rgb_fname+"*"))
    if not reproduce:
        rgb_file = random.sample(rgb_list, 1)[0]
    else:
        rgb_file = rgb_list[0]
    print("batch chosen: ", rgb_file[len(rgb_dir+rgb_fname):-2])
    
    rgb_dic = np.load(rgb_file, allow_pickle=True)
    
    if not reproduce:
        img_index = random.sample(range(0, rgb_dic["normal"].shape[0]), 2)
    else:
        img_index = [0,1]
    print("img_indexes: ", img_index)

    rgb_normal = rgb_dic["normal"][img_index]*Stds+Means
    rgb_noise_capped = rgb_dic["noise_capped"][img_index]*Stds+Means
    reprod = "reprod" if reproduce else ""

    col_titles = ["Normal noise", "Noise capped", "Abs difference"]*2
    suptitle = f"Random samples comparison with and without noise capped to {noise_max}"
    pic_name = f"noise_v_no_noise_{reprod}_dom_{dom_size}_lat_{lat}_bs_{bs}_chm_{chm}_vars"+\
            f"_{'_'.join(v for v in vars)}_noise_{use_noise}_max_{noise_max}_ckpt_{ckpt}.png"

    data = []
    for i in range(rgb_normal.shape[0]):
        data.append(rgb_normal[i])
        data.append(rgb_noise_capped[i])
        data.append(np.abs(rgb_normal[i]-rgb_noise_capped[i]))
        #print("Number of pixels that are different: ", np.count_nonzero(rgb_noise_capped[i]!=rgb_normal[i]))
        #assert(rgb_noise_capped[i,0,0,0]!=rgb_normal[i,0,0,0]), "Houston we've got a problem"

    data = np.array(data)

    can = art.canvasHolder("SE_for_GAN" if dom_size==128 else "SE_GAN_extend", dom_size, dom_size)
    var_names = [(var, var_dict_unit[var]) for var in vars]
    can.plot_abs_error_sev_cbar(data=data, var_names=var_names, suptitle=suptitle,
                                plot_dir=Filepath + "rgb_no_noise/", pic_name=pic_name, 
                                col_titles=col_titles, cmap_wind="viridis", cmap_t="rainbow")


if __name__=="__main__":
    lat = 512
    chm = 2
    bs = 16
    ckpt = 147000
    vars = ['u','v','t2m','z500','t850','tpw850']
    noise = True
    dom_size = 256
    noise_max = 1e-7
    scale = [1,2,3,4,5]
    new_weights = [0]
    reproduce = True

    if torch.cuda.is_available():
        torch.manual_seed(73)
        print("Creating samples ...")
        fpath = "/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/"
        #samples_without_noise_influence(lat=lat, bs=bs, chm=chm, lr=0.002, ckpt=ckpt, noise_max=noise_max,
        #                            vars=vars, dom_size=dom_size, use_noise=noise, fpath=fpath)
        modify_noise_per_scale(lat=lat, bs=bs, chm=chm, lr=0.002, ckpt=ckpt, vars=vars, dom_size=dom_size,
                    use_noise=noise, fpath=fpath, scale=scale, new_weights=new_weights)
    
    else:
        print("Plotting ...")
        fpath = "/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/"
        #plot_rgb_noise_v_no_noise(lat=lat, bs=bs, chm=chm, lr=0.002, ckpt=ckpt, 
        #                            vars=vars, dom_size=dom_size, use_noise=noise, fpath=None)
        plot_noise_change_scale(lat=lat, bs=bs, chm=chm, lr=0.002, ckpt=ckpt, vars=vars, dom_size=dom_size,
                    use_noise=noise, fpath=fpath, scale=scale, new_weights=new_weights, reproduce=reproduce)
