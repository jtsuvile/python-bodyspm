from bodyfunctions import *
import h5py
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import re

# correlating maps on a subject level, combining to an array

# FIBRO seems to have a bigger impact on the maps than others, is this because fibro affects more of the body?
# TODO: also try with [0,1] maps, not just [-1,1]

figloc = '/m/nbe/scratch/socbrain/kipupotilaat/figures/'
maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'
who = 'stockholm'

if who == 'patients':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/'
elif who == 'stockholm':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/stockholm/processed/'
elif who=='matched_controls':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/matched_controls/'
elif who=='controls':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/'
else:
    sys.exit(0)

datafile = get_latest_datafile(dataloc)

mask_fb = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')
mask_one = read_in_mask(maskloc + 'mask_front_new.png')

stim_names = {'emotions_0':'sadness', 'emotions_1':'happiness', 'emotions_2':'anger', 'emotions_3':'surprise',
              'emotions_4': 'fear', 'emotions_5':'disgust', 'emotions_6':'neutral', 'pain_0':'acute pain',
              'pain_1': 'chonic_pain', 'sensitivity_0': 'tactile sensitivity',
              'sensitivity_1': 'nociceptive sensitivity', 'sensitivity_2': 'hedonic sensitivity'}

order_maps = []
cond = 'pain_1'
cond2 = 'emotions_2'

for j, cond2 in enumerate(stim_names.keys()):
    order_maps.append(cond)
    print(cond2)
    with h5py.File(datafile, 'r') as p:
        data = p[cond].value
        data2 = p[cond2].value
        subids = p['subid'].value
        kipu_diagnoses = list(p['groups'])
        interesting_indices = np.asarray([x == 'FIBROMYALGI' for x in kipu_diagnoses])
        data = data[interesting_indices, :, :]
        data2 = data2[interesting_indices, :, :]

    dims = data.shape
    dims2 = data2.shape

    if j == 0:
        res_maps_r = np.zeros((dims2[0], len(stim_names.keys())))
        res_maps_p = np.zeros((dims2[0], len(stim_names.keys())))
    if dims[2]==dims2[2]:
        print('map dimensions are a match.')
    elif dims[2] == 2 * dims2[2]:
        print('map1 smaller than map2, converting..')
        data_left = data[:,:,0:171]
        data_right = data[:,:,171:342]
        data_right_rotated = np.fliplr(data_right)
        data_new = np.add(data_left, data_right_rotated)
        data = data_new
    else:
        print('map dimensions not a match.')

    for sub in range(0, dims2[0]):
        #print(sub)
        r_res = np.corrcoef(data[sub, :, :].flatten(order='C'), data2[sub, :, :].flatten(order='C'))
        res_maps_r[sub,j] = r_res[0,1]

print(res_maps_r)
np.nanmean(res_maps_r, axis=0)

#Visualise results

# fig = plt.figure()
#
# ind=0
# v=0
# pattern = re.compile("emotions_.")
# if pattern.search(cond_2):
#     relevant_mask = np.hstack((mask_one, np.zeros((522, 171))))
# else:
#     relevant_mask = mask_fb
#
# temp_data_2 = res_maps_r[ind, :, :]
# # fixed_p, p_reject = p_adj_maps(res_maps_p[ind, :, 0:342], mask=mask_fb, method='fdr_bh')
# # temp_data[fixed_p > 0.05] = 0
# # temp_data_2[res_maps_p[ind, :, :] > 0.05] = 0
# masked_data_2 = np.ma.masked_where(relevant_mask != 1, temp_data_2)
# im2 = plt.imshow(masked_data_2, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title('spatial correlation between '+ stim_names[cond]+' and '+ stim_names[cond_2])
# plt.axis('off')
#
# fig.colorbar(im2)
# plt.savefig(figloc+'CRPS_correlation_'+cond+'_'+cond_2+'_no_fdr.png')
# plt.close()

#
# fig3, axs3 = plt.subplots(1, 1, figsize=(15, 6), facecolor='w', sharex=True, sharey=True)
#
# ind=0
# v=0
# temp_data_2 = mask_fb[:, :]
# fixed_p, p_reject = p_adj_maps(mask_fb[:, 0:342], mask=mask_fb, method='fdr_bh')
# #temp_data[fixed_p > 0.05] = 0
# # temp_data_2[res_maps_p[ind, :, :] > 0.05] = 0
# masked_data_2 = np.ma.masked_where(mask_fb != 1, temp_data_2)
# im3 = axs3.imshow(masked_data_2, cmap='coolwarm', vmin=-0.8, vmax=0.8)
#
# fig3.colorbar(im3)
# plt.savefig(figloc+'vis_mask_corr.png')
# plt.close()
