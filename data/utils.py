import pandas as pd 
import numpy as np

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

def load_batch_from_timestamp(dataframe, date, lt, data_dir, Shape=(3,256,256), var_indices=[0,1,2], crop_indices = [0,256,0,256]):

    df0 = dataframe[(dataframe['Date']==date) & (dataframe['LeadTime']==lt)]

    Nb = len(df0)

    batch = np.zeros((Nb,) + tuple(Shape))
    print(batch.shape)
    for i,s in enumerate(df0['Name']):

        sn = np.load(f'{data_dir}{s}.npy')[var_indices,crop_indices[0]:crop_indices[1],crop_indices[2]:crop_indices[3]].astype(np.float32)

        batch[i] = sn
    
    return batch

def rescale(generated, Mean, Max, scale) : 
    
    return scale * Max * generated + Mean

def scale(input, Mean, Max, scale) : 
    
    return scale * (input - Mean) / Max
