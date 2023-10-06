import numpy as np

data_dir = ''
real_data_dir = ''

const_mean_real = ''
const_max_real = ''

const_mean_fake = ''
const_max_fake = ''
for i in range(74):
    for j in range(8):

        real = np.load(data_dir + f'Rsemble_{float(i)}_{float(j)}.npy')
        fake = np.load(data_dir + f'Rsemble_{float(i)}_{float(j)}.npy')
