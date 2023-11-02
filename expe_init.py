#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:43:56 2022

@author: brochetc

WARNING : this file interfaces several processes
     ---> experiment files configuration (handled by the main script here)
     ---> experiment launching (handled by slurm)
     ---> tracking of time-outs and relaunch (handled by checkpoint_manager.sh)
     Its output is therefore recorded, 
     as well as it records outputs from other
     files

"""
import os
from glob import glob
import argparse
from itertools import product
import subprocess
import yaml
from pathlib import Path


def check_file(file):
    # Search/download file (if necessary) and return path
    file = str(file)  # convert to str()
    if Path(file).is_file() or file == '':  # exists
        return file
    elif file.startswith(('http://', 'https://')):  # download
        url, file = file, Path(urllib.parse.unquote(str(file))).name  # url, file (decode '%2F' to '/' etc.)
        file = file.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, file)
        assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        return file
    else:  # search
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file

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

def str2inttuple(li):
    if type(li)==list:
        li2 =[int(p) for p in li]  
        return tuple(li2)
    
    elif type(li)==str:
        li2 = li[1:-1].split(',')
        li3 =[int(p) for p in li2]

        return tuple(li3)

    else : 
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))

def str2VQparams(str):
    """
    convert string to dictionary
    """

    dico = {}

    return dico


# very_small_exp = True
# small_exp = False
# normal_exp_steps = 150000

# ensemble={'--batch_size':[16], 
#           '--lr_D': [0.002],'--lr_G' : [0.002],
#           '--g_channels':[3], '--d_channels':[3],
#           '--var_names' : ["['u','v','t2m','z500','t850','tpw850']"], 
#           '--total_steps' :[20 if very_small_exp else (1000 if small_exp else normal_exp_steps)], # 'rr', 'orog' ['u','v','t2m']
#           '--conditional':[False],
#           '--path_batch_shrink' : [2],
#           '--model' :['stylegan2'], '--train_type' : ['stylegan'],
#           '--latent_dim' : [512],
#           '--pretrained_model' : [0],
#           '--use_noise' : [True],
#           #'--crop_indexes' : ["[0,256,0,256]"],
#           '--crop_indexes' : ["[78,206,55,183]"]
# }

# monovar = ("['u','v','t2m']" not in ensemble['--var_names'])
# use_noise = ensemble['--use_noise'][0]
# all_domain = True if sum(str2intlist(ensemble['--crop_indexes'][0])[1::2])-\
#                         sum(str2intlist(ensemble['--crop_indexes'][0])[0::2])==512 else False

def create_new(keyword):
    """
    create new directory with given keyword, when there are already other directories with
    this keyword
    
    """
    previous=len(glob('*'+keyword+'*'))
    INSTANCE_NUM=previous+1
    os.mkdir(keyword+'_'+str(INSTANCE_NUM))

class AttrDict(object):
    def __init__(self, _dict):
        self.__dict__.update(_dict)


def read_yamlconfig(config_file_abs_path) : 

    print("config_file_abs_path", config_file_abs_path)

    if Path(config_file_abs_path).is_file() :  # exists
        with open(config_file_abs_path) as f: 
            optyaml = yaml.safe_load(f)  # load hyps

    very_small_exp = optyaml["very_small_exp"]

    small_exp = optyaml["small_exp"]
    normal_exp_steps = optyaml["normal_exp_steps"]
    ensemble = optyaml["ensemble"]
    monovar = ("['u','v','t2m']" not in ensemble['--var_names'])
    use_noise = ensemble['--use_noise'][0]
    all_domain = True if sum(str2intlist(ensemble['--crop_indexes'][0])[1::2])-\
                            sum(str2intlist(ensemble['--crop_indexes'][0])[0::2])==512 else False
    VQparams = optyaml["VQparams_file"]

    return  optyaml, very_small_exp, small_exp, normal_exp_steps, ensemble, monovar, use_noise, all_domain , config_file_abs_path

def get_dirs(home_dir):

    
    """
    
    read config file for the experimentations
    
    """
    
    parser=argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, \
                        default='configs/very_small_exp_256.yaml')
    args =   parser.parse_args()
    config_file_abs_path = home_dir + "/" + args.config_file 

    return read_yamlconfig(config_file_abs_path)



    # parser.add_argument('--data_dir', type=str, \
    #                     default='/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_1024_done/')
    #                     # '/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_done/'
    # parser.add_argument('--output_dir', type=str, \
    #                     default='/scratch/mrmn/reganc/Exp_StyleGAN/Set_2')
    

    # parser.add_argument('--mean_file', type=str, default='mean_with_8_var.npy')
    # parser.add_argument('--max_file', type=str, default='max_with_8_var.npy')
    # parser.add_argument('--id_file', type=str, default='IS_method_labels_8_var.csv')

    # parser.add_argument('--SET_NUM', type=int, \
    #                 default=2) # run test jobs in Set_2
    
    # parser.add_argument('--ensemble', type=str, \
    #                     default='configs/very_small_exp.yaml')

    # # using pretrained model modifies the behavior of dir creation
    # parser.add_argument('--pretrained_model', type=int, default=0,\
    #                     help='step at which pretrained model have been saved')
    
    # parser.add_argument('--which_experiment',type=int, default=-1,\
    #                     help='index of the experiment to run if pretrained_model is activated') # not really used anymore
    
    # parser.add_argument('--max_relaunch', type=int, default=0, 
    #                     help ='number of times we relaunch the expe after timeout')

    # parser.add_argument('--auto_relaunch', type=str2bool, default=False,
    #                     help='set to Yes if you want an automatic relaunch. Manual relaunch (e.g. from pretrained)\
    #                     should be passed with auto_relaunch=False to prevent overwriting previous results.')

    # parser.add_argument('--slurm_file', type=str, default='run_GAN.slurm',\
    #                     help='slurm file in slurm_dir')


def get_expe_parameters():

    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--data_dir', type=str,default="/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_done/" )
    parser.add_argument('--mean_file', type=str,default="mean_with_8_var.npy" )
    parser.add_argument('--max_file', type=str,default="max_with_8_var.npy" )
    parser.add_argument('--id_file_train', type=str, default="IS_method_labels_8_var.csv" )
    parser.add_argument('--id_file_test', type=str, default="IS_method_labels_8_var.csv" )
    parser.add_argument('--pretrained_model', type=int, default=0)

    # if not dirs["interactive"] : 
        # change output_dir default path for experiments set
    parser.add_argument('--output_dir', type=str, \
                        default=os.getcwd()+'/')
    # else : 
    #     parser.add_argument('--output_dir', type=str, \
    #                         default=dirs["output_dir"] + "gpu_play")


    # Model architecture hyper-parameters
    
    parser.add_argument('--model', type=str, default='stylegan2', \
                        choices=['resnet','resnet_aa' ,'resnet_concat', 'resnet_concat_aa', 'resnet_concat_smooth', 
                                 'resnet_concat_bilin', 'resnet_concat_light', 'cBN_proj','cBN_proj_simple',
                                 'resnet_concat_bilin_emb', 'resnet_concat_rough_emb', 'stylegan2', 'stylegan2_fp16','swagan'])
    
    
    
    # choices of loss function and initialization
    parser.add_argument('--train_type', type=str, default='stylegan',\
                        choices=['vanilla','wgan-gp', 'wgan-hinge','stylegan'])
    
    
    # conditional arguments
    parser.add_argument('--conditional', type=str2bool, default = True)
    parser.add_argument('--condition_vars', type=str2list, default = ['u','v', 't2m'])
    parser.add_argument('--lambda_start_step', type=int, default=2000)
    parser.add_argument('--lambda_stop_step', type=int, default=4000)
    parser.add_argument('--discrete_level', type=int, default=0)
    parser.add_argument('--mlp_inj_level', type=int, default=1)
    parser.add_argument('--VQparams_file', type=str, default=None)
    
    #architectural choices
    
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--g_channels', type=int, default=3)
    parser.add_argument('--d_channels', type=int, default=3)
    parser.add_argument('--n_mlp', type=int, default=8, help="depth of the z->w mlp")
    parser.add_argument("--channel_multiplier",type=int, default=2,
        help="channel multiplier factor for the stylegan/swagan model. config-f = 2, else = 1",
    )
    # regularisation settings(wasserstein GAN)

    parser.add_argument('--lamda_gp', type=float, default=10.0)
    parser.add_argument('--ortho_init',type=str2bool, default=False)
    
    parser.add_argument('--sn_on_g', type=str2bool, default=False,\
                        help='Apply spectral normalisation on Generator')
    
    # regularisation settings (styleGAN)
    
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--path_regularize",type=float,default=2,\
                        help="weight of the path length regularization")

    parser.add_argument( "--path_batch_shrink",type=int,default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)")
    
    
    parser.add_argument("--d_reg_every",type=int,default=16,
                        help="interval of the applying r1 regularization")
    
    parser.add_argument("--g_reg_every",type=int, default=4,
        help="interval of the applying path length regularization")
    
    parser.add_argument("--mixing", type=float, default=0.9, 
                        help="probability of latent code mixing")
    
    # augmentation and ADA settings (styleGAN)
    
    parser.add_argument("--augment", action="store_true", 
                        help="apply non leaking augmentation"
    )
    parser.add_argument("--augment_p", type=float, default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument("--ada_target",type=float,default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument("--ada_length",type=int, default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument("--ada_every", type=int,default=256,
                        help="probability update interval of the adaptive augmentation",
    )

    # Training settings
    parser.add_argument('--epochs_num', type=int, default=1,\
                        help='how many times to go through dataset')
    parser.add_argument('--total_steps', type=int, default=50000,\
                        help='how many times to update the generator')
    
    parser.add_argument('--batch_size', type=int, default=16)

    
    parser.add_argument('--lr_G', type=float, default=0.002)
    parser.add_argument('--lr_D', type=float, default=0.002)
    
    parser.add_argument('--beta1_D', type=float, default=0.0)
    parser.add_argument('--beta2_D', type=float, default=0.9)
    
    parser.add_argument('--beta1_G', type=float, default=0.0)
    parser.add_argument('--beta2_G', type=float, default=0.9)
    
    parser.add_argument('--warmup', type=str2bool, default=False)
    parser.add_argument('--use_noise', type=bool, default=True, help="if False, doesn't use noise_inj")
    
    # Data description
    parser.add_argument('--var_names', type=str2list, default=['u','v','t2m'])#, 'orog'])
    parser.add_argument('--crop_indexes', type=str2intlist, default=[78,206,55,183])

    parser.add_argument('--crop_size', type=str2inttuple, default=(128,128) ) #   if not all_domain else (256,256))
    parser.add_argument('--full_size', type=str2inttuple, default=(256,256))
    
    # Training settings -schedulers
    parser.add_argument('--lrD_sched', type=str, default='None', \
                        choices=['None','exp', 'linear'])
    parser.add_argument('--lrG_sched', type=str, default='None', \
                        choices=['None','exp', 'linear'])
    parser.add_argument('--lrD_gamma', type=float, default=0.95)
    parser.add_argument('--lrG_gamma', type=float, default=0.95)
    
    
    # Testing and plotting setting
    parser.add_argument('--test_samples',type=int, default=4 ) # if all_domain else 256,help='samples to be tested')
    parser.add_argument('--plot_samples', type=int, default=16)
    parser.add_argument('--sample_num', type=int, default=16, help='Samples to be saved') #  if all_domain else 256,\
    


    # Misc
    parser.add_argument('--fp16_resolution', type=int, default=1000) # 1000 --> not used
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

    # Step size

    parser.add_argument('--log_epoch', type=int,
                        default=1)
    parser.add_argument('--sample_epoch', type=int,
                        default=1)
    parser.add_argument('--plot_epoch', type=int,
                        default=1)
    parser.add_argument('--save_epoch', type=int,
                        default=1)
    parser.add_argument('--test_epoch', type=int,
                        default=1)
    parser.add_argument('--log_step', type=int, default=5)# if very_small_exp else (1000 if small_exp else 3000)) #-> default is at the end of each epoch
    parser.add_argument('--sample_step', type=int, default=5)# if very_small_exp else (1000 if small_exp else 3000)) # set to 0 if not needed
    parser.add_argument('--plot_step', type=int, default=5)# if very_small_exp else (1000 if small_exp else 3000)) #set to 0 if not needed
    parser.add_argument('--save_step', type=int, default=5)# if very_small_exp else (1000 if small_exp else 3000)) # set to 0 if not needed
    parser.add_argument('--test_step', type=int, default=5)# if very_small_exp else (1000 if small_exp else 3000)) #set to 0 if not needed
       
    return parser

def make_dicts(ensemble,  option='cartesian'):
    """
    make cartesian product of parameters used in ensemble
    
    input :
        ensemble : dict of shape { parameter : list(parameters to test) }
        
    output :
        cross product of ensemble as list of dictionaries, each entry of shape
        {parameter : value}
    """

    allowed_args=set(vars(get_expe_parameters())['_option_string_actions'])
    keys=set(ensemble)
    if keys - allowed_args != set():
        raise ValueError(f"Some args in ensemble are unexpected : {keys - allowed_args}")
    prod_list=[]
    list_items = list(ensemble.values())
    if option=='cartesian':
        for item in product(*(list_items)):
            prod_list.append(dict((key[2:],i) for key, i in zip(ensemble.keys(),item)))
    return prod_list

def sanity_check(config) :
    """
    Perform verifications on Namespace config and adds some fields to it
    """
    config.g_output_dim = config.crop_size
    config.d_input_dim = config.crop_size
    
    H, W = config.crop_indexes[1]-config.crop_indexes[0], config.crop_indexes[3]-config.crop_indexes[2]

    assert (H,W)==config.d_input_dim
    
    assert ((config.test_step==0) or (config.log_step%config.test_step==0))
    
    return config

def nameSpace2SlurmArg(config):
    """
    transform an argparse.namespace to slurm-readable chain of args
    
    input :
        config  -> an argParse.Namespace as created by argparse.parse_args()
    
    output :
        a chain of characters summing up the namespace
    """
    
    dic=vars(config)
    li=[]
    for key in dic.keys():
        if key=='var_names':
            print(dic[key])
        value=dic[key]
        li.append('--'+key+'='+str(value))

    return "|".join(li)

def get_slurm_file(bytecode) : 
    """
    parse a bytecode stdout to extract the name of slurm file
    optionnally, write this name to Memo_readme.csv
    
    """
    ind0 = sbatch_output.stdout.find(b'Submitted')
        
    if ind0>=0 :
        ind1 = sbatch_output.stdout.find(b'job ')
        slurm_file = 'slurm-' + sbatch_output.stdout[ind1+4:-1].decode('utf-8') + '.out'
        slurm_file_num = int(sbatch_output.stdout[ind1+4:-1].decode('utf-8'))
        
    else :
        slurm_file = None
        slurm_file_num = None
    return slurm_file, slurm_file_num

def prepare_expe(config, use_pretrained = False):
    """
    create output folders in the current directory 
    for a single experiment instance
    
    input :
        
        a config namespace as generated by get_expe_parameters
    
    ouput : 
        NAME : the name of the main experiment directory
    """
    NAME = config.model+'_'+config.train_type+f'_dom_{config.crop_size[0]}_lat-dim_'+str(config.latent_dim)+'_bs_'\
    +str(config.batch_size)+'_'+str(config.lr_D)+'_'+str(config.lr_G)+'_ch-mul_'+str(config.channel_multiplier)+\
        '_vars_'+'_'.join(str(var_name[1:-1]) for var_name in config.var_names)+f'_noise_{use_noise}'
    
    print(NAME)
    
    base_dir = os.getcwd()
    if not os.path.exists(NAME):
        os.mkdir(NAME)
    os.chdir(NAME)
    
    INSTANCE_NUM = len(glob('*Instance*'))
    
    if not use_pretrained :
        INSTANCE_NUM += 1
        os.mkdir('Instance_'+str(INSTANCE_NUM))
    
    print('Instance num is {}'.format(INSTANCE_NUM))

    os.chdir('Instance_'+str(INSTANCE_NUM))
    expe_dir=os.getcwd()
    
    with open('ReadMe_'+str(INSTANCE_NUM)+'.txt', 'a') as f:
        f.write('-----------------------------------------\n')
        for arg in config.__dict__.keys():
            f.write(arg+'\t:\t'+str(config.__dict__[arg])+'\n')
        f.close
    if not os.path.exists('log'):
        os.mkdir('log')
        
    if not os.path.exists('models'):
        os.mkdir('models')
        
    if not os.path.exists('samples'):
        os.mkdir('samples')
    
    os.chdir(base_dir)
    return NAME, expe_dir

def parse_Memo(where, experiment, pretrained = None):
    
    base_dir = where
    
    with open(base_dir+'/Memo_readme.csv','r') as f :

        lines = f.readlines()
    
        # retrieving last line of parameters 
        n = 1
        param_names=[32]
        
        while param_names[0]!='batch_size' :
            param_names = lines[experiment-n].split(',')
            n+=1
        print(param_names)
        param_names[-1] = param_names[-1][:-1]
        print(param_names)
        
        print(lines[experiment])
        ind0 = lines[experiment].find('[')
        ind1 = lines[experiment].find(']')
        param_values_0 = lines[experiment][:ind0-1].split(',')
        param_values_1 = lines[experiment][ind1+2:].split(',')
        
        list_var = lines[experiment][ind0:ind1+1]
        
        param_values = param_values_0 + [list_var] + param_values_1
        
        param_values[-1] = param_values[-1][:-1]
        
    params = { k : v for k,v in zip(param_names, param_values)}
    
    if pretrained is not None :
        
        params["pretrained_model"] = pretrained
    
    return params
    

def prepare_expe_set(where, expe_list, Instance = None):
    """
    prepare a set of experiments through use of cross product namespaces
    
    input :
        list of dicts containing each experiment parameters
    
    output :
        list of namespaces containing each experiment parameters
    """
    
    
    base_dir = where
    config_list=[]
    with open('Memo_readme.csv', 'a') as file:
        
        strKeys=list(expe_list[0].keys())
        
        file.write(','.join(strKeys)+'\n')

    for params in expe_list:
        
        args = ['--'+k+'='+str(v) for k,v in params.items()]

        # print("\n\n\n ************* prepare_expe_set")

        config = get_expe_parameters().parse_args(args=args)
        config = sanity_check(config)
        
        NAME, expe_dir = prepare_expe(config,
                                     use_pretrained = (config.pretrained_model>0))
        
        config.output_dir = expe_dir
        config_list.append(config)
        params["output_dir"] = expe_dir
        
        
        #writing in memo file
        strVals={key : str(value) for key, value in params.items()}
        
        with open('Memo_readme.csv', 'a') as file:
            file.write(','.join(strVals.values())+'\n')

    os.chdir(base_dir)
    
    return config_list


if __name__=="__main__":


    # slurm_dir = os.path.dirname(os.path.realpath(__file__)) +'/slurms/'
    # ckpt_dir = os.path.dirname(os.path.realpath(__file__))
    # print('ckpt_dir', ckpt_dir)
    # dirs = get_dirs()
    # previous_sets = len(glob(dirs.output_dir+'*Set*'))

    home_dir = os.path.dirname(os.path.realpath(__file__))
    slurm_dir = os.path.dirname(os.path.realpath(__file__)) +'/slurms/'

    dirs, very_small_exp, small_exp, normal_exp_steps, ensemble, monovar, use_noise, all_domain, config_file_abs_path = get_dirs(home_dir)
    dirs = AttrDict(dirs)


    # if Path(ensemble_file).is_file() :  # exists
    #     with open(ensemble_file) as f: 
    #         optyaml = yaml.safe_load(f)  # load hyps

    # very_small_exp = optyaml["very_small_exp"]
    # small_exp = optyaml["small_exp"]
    # normal_exp_steps = optyaml["normal_exp_steps"]
    # ensemble = optyaml["ensemble"] 

    # monovar = ("['u','v','t2m']" not in ensemble['--var_names'])
    # use_noise = ensemble['--use_noise'][0]
    # all_domain = True if sum(str2intlist(ensemble['--crop_indexes'][0])[1::2])-\
    #                         sum(str2intlist(ensemble['--crop_indexes'][0])[0::2])==512 else False

    previous_sets = len(glob(dirs.output_dir+'*Set*'))
    
    SET_NUM = dirs.SET_NUM #2 if very_small_exp or small_exp else \
    #        5 if all_domain else\
    #        3 if monovar else\
    #        4 if not use_noise else\
    #        1
    
    # je suis pas méga sûr de ce qui suit donc je commente pour garder mon arborescence
    #if dirs.SET_NUM>0:
    #    SET_NUM = dirs.SET_NUM
    #else:    
    #    SET_NUM = previous_sets+1
    where = dirs.output_dir+'Set_'+str(SET_NUM)
    
    if not os.path.exists(where):
        os.mkdir(where)
    os.chdir(where)
        
    ensemble['--data_dir']=[dirs.data_dir]
    ensemble['--output_dir']=[dirs.output_dir]
    ensemble['--mean_file']=[dirs.mean_file]
    ensemble['--max_file']=[dirs.max_file]
    ensemble['--id_file_train']=[dirs.id_file_train]
    ensemble['--id_file_test']=[dirs.id_file_test]
    # normalement même pas obligé de mettre cette ligne vu qu'il est déjà dans ensemble
    ensemble['--pretrained_model']=dirs.ensemble["--pretrained_model"]
    ensemble['--VQparams_file']=[dirs.VQparams_file]
    
    if not dirs.auto_relaunch:
        expe_list = make_dicts(ensemble)
        
        config_list = prepare_expe_set(where, expe_list)
    
        
    else :
        
        expe_list = [ parse_Memo(where, dirs.which_experiment,
                                pretrained=dirs.pretrained_model) ]
    
        config_list = prepare_expe_set(where, expe_list)
    
          
    for i, params in enumerate(expe_list):
        print('--------------------------------------------------------------')
        print('--------------------------------------------------------------')
        print('Running experiment')
        print('--------------------------------------------------------------')
        print('--------------------------------------------------------------')
        
        args = nameSpace2SlurmArg(argparse.Namespace(**params))
        
        print('\n\n--------------------------------------------------------------')
        print("args", args)
        print('--------------------------------------------------------------')

        env_slurm ={**os.environ,
            "HOME_DIR":home_dir,
            "OUTPUT_DIR":dirs.output_dir, 
            "DATA_DIR":dirs.data_dir,
            "SLURM_SCRIPT":home_dir + "/" + dirs.slurm_docker_run_perso,
            "PYTHON_SCRIPT":home_dir +  "/"+ dirs.main_file, 
            "CONFIG_FILE": config_file_abs_path,
        } 
        print("env_slurm SLURM_SCRIPT", env_slurm["SLURM_SCRIPT"])
        print("env_slurm PYTHON_SCRIPT", env_slurm["PYTHON_SCRIPT"])
        
        sbatch_output = subprocess.run(['sbatch',slurm_dir+dirs.slurm_file, args], capture_output=True,  env=env_slurm)
        
        slurm_file, slurm_file_num = get_slurm_file(sbatch_output)
        
        slurm_file = where +'/'+ slurm_file 
        
        print(slurm_file) # this line is crucial ! do not delete !!!!!
         
        # this part to automatically launch checkpoint_manager on the experiment
        # run it as a background task
        # and write output in checkpoint_log file
        timeout = 12 * 3600 + 600 
        
        if  slurm_file is not None and dirs.max_relaunch>0:

            output_file = where+'/checkpoint_log_{}.txt'.format(slurm_file_num)

            f  = open(output_file, 'w')
            f.write('Checkpoint loading\n')
            f.close()
            
            #f = open(output_file,'a')
            #subprocess.Popen(['bash',home_dir+'/checkpoints_manager.sh',
            #                config_list[i].output_dir,slurm_file,str(dirs.SET_NUM),str(dirs.max_relaunch),str(timeout)], stdout=f)
            with open(output_file,'a') as f :
                subprocess.run(['bash',home_dir+'/checkpoints_manager.sh',
                config_list[i].output_dir,slurm_file,str(SET_NUM),str(dirs.max_relaunch),str(timeout)],timeout=timeout, stdout=f)
            # j'ai changé de dirs.set_num à set_num encore une fois pour garder mon arboresecnce -> à discuter pour next réu
