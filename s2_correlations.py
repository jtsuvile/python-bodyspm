from bodyfunctions import *
import h5py
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

import sys

figloc = '/m/nbe/scratch/socbrain/kipupotilaat/figures/'
maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'
dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/'
datafile = get_latest_datafile(dataloc)
dataloc_controls = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed'
datafile_controls = get_latest_datafile(dataloc_controls)

pain_duration = pd.read_csv('/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/pain_start_helsinki_summer2019.csv',
                            delimiter=';')

with h5py.File(datafile, 'r') as p:
    subids = p['subid'].value


mask_fb = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')
mask_one = read_in_mask(maskloc + 'mask_front_new.png')

stim_names = {'emotions_0': ['sadness', 0], 'emotions_1': ['happiness', 0], 'emotions_2': ['anger', 0],
                       'emotions_3': ['surprise', 0], 'emotions_4': ['fear', 0], 'emotions_5': ['disgust', 0],
                       'emotions_6': ['neutral', 0], 'pain_0': ['acute pain', 1], 'pain_1': ['chonic pain', 1],
                       'sensitivity_0': ['tactile sensitivity', 1],
                       'sensitivity_1': ['nociceptive sensitivity', 1], 'sensitivity_2': ['hedonic sensitivity', 1]}

stim_names_emotions = {'emotions_0':'sadness', 'emotions_1':'happiness', 'emotions_2':'anger', 'emotions_3':'surprise',
              'emotions_4': 'fear', 'emotions_5':'disgust', 'emotions_6':'neutral'}

stim_names_pain = {'pain_0':'acute pain', 'pain_1': 'chonic_pain'}
stim_names_sensitivity = {'sensitivity_0':'tactile sensitivity',
              'sensitivity_1':'nociceptive sensitivity', 'sensitivity_2':'hedonic sensitivity'}

res_maps_r = np.zeros((len(stim_names.keys()), 522, 342))
res_maps_p = np.zeros((len(stim_names.keys()), 522, 342))

res_maps_r_bpi = np.zeros((len(stim_names.keys()), 522, 342))
res_maps_p_bpi = np.zeros((len(stim_names.keys()), 522, 342))

order_maps = []

pain_duration_ordered = []

for subject in subids:
    sub_ind = np.where(pain_duration['subid']==subject)[0][0]
    pain_start_sub = pain_duration['pain_Start'][sub_ind]
    pain_duration_ordered.append(pain_start_sub)


#corr_with_this = 'feels_pain'
also_bpi = False
corr_with = pain_duration_ordered
corr_with_this = 'pain_duration'



bpi = ['bpi_worst', 'bpi_least', 'bpi_average', 'bpi_now']
for j, cond in enumerate(stim_names_emotions.keys()):
    order_maps.append(cond)
    with h5py.File(datafile, 'r') as p:
        data = p[cond].value
        subids = p['subid'].value
        #corr_with = p[corr_with_this].value
        if also_bpi:
            bpi_worst = p['bpi_worst'].value
            bpi_worst[bpi_worst < 0] = 0
            bpi_least = p['bpi_least'].value
            bpi_least[bpi_least < 0] = 0
            bpi_average = p['bpi_average'].value
            bpi_average[bpi_average < 0] = 0
            bpi_now = p['bpi_now'].value
            bpi_now[bpi_now < 0] = 0
            bpi_composite = np.average(list(zip(bpi_worst, bpi_least, bpi_average, bpi_now)), axis=1)

    # if showing diff between pain patients & controls:
    # with h5py.File(datafile_controls, 'r') as c:
    #     control = c[cond].value
    #     corr_with = c[corr_with_this].value

    dims = data.shape
    ## for two-sides data, binarize and run point beserial correlation, which with scipy gives same results as
    # scipy.stats.pearsonr
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pointbiserialr.html
    data_binary = binarize(data)
    # temporarily change data to 2-D to enable correlation analysis
    data_reshaped = np.reshape(data_binary, (dims[0], -1))

    if cond not in stim_names_emotions.keys():
        corr_res_pearson, corr_p_pearson = np.apply_along_axis(stats.pearsonr, 0, data_reshaped, corr_with)
        # reshape result
        corr_map_pearson = np.reshape(corr_res_pearson, (dims[1], dims[2]))
        p_map_pearson = np.reshape(corr_p_pearson, (dims[1], dims[2]))
        res_maps_r[j, :, :] = corr_map_pearson
        res_maps_p[j, :, :] = p_map_pearson
        if also_bpi:
            corr_res_pearson_bpi, corr_p_pearson_bpi = np.apply_along_axis(stats.pearsonr, 0, data_reshaped, bpi_composite)
            corr_map_pearson_bpi = np.reshape(corr_res_pearson_bpi, (dims[1], dims[2]))
            p_map_pearson_bpi = np.reshape(corr_p_pearson_bpi, (dims[1], dims[2]))
            res_maps_r_bpi[j, :, :] = corr_map_pearson_bpi
            res_maps_p_bpi[j, :, :] = p_map_pearson_bpi
    else:
        print('cond: ', cond, 'using spearman.')
        corr_res_spearman, corr_p_spearman = np.apply_along_axis(stats.spearmanr, 0, data_reshaped, corr_with)
        # reshape result
        corr_map_spearman = np.reshape(corr_res_spearman, (dims[1], dims[2]))
        p_map_spearman = np.reshape(corr_p_spearman, (dims[1], dims[2]))
        res_maps_r[j, :, 0:171] = corr_map_spearman
        res_maps_p[j, :, 0:171] = p_map_spearman
        if also_bpi:
            corr_res_spearman_bpi, corr_p_spearman_bpi = np.apply_along_axis(stats.spearmanr, 0, data_reshaped, bpi_composite)
            corr_map_spearman_bpi = np.reshape(corr_res_spearman_bpi, (dims[1], dims[2]))
            p_map_spearman_bpi = np.reshape(corr_p_spearman_bpi, (dims[1], dims[2]))
            res_maps_r_bpi[j, :, 0:171] = corr_map_spearman_bpi
            res_maps_p_bpi[j, :, 0:171] = p_map_spearman_bpi

# test success with plotting
## PAIN
#
#
# fig2, axs2 = plt.subplots(1, 2, figsize=(15, 6), facecolor='w', sharex=True, sharey=True)
# fig2.subplots_adjust(hspace= 0 , wspace= 0 )
#
# axs2 = axs2.ravel()
#
# for v, sense in enumerate(stim_names_pain.keys()):
#     ind = order_maps.index(sense)
#     temp_data_2 = res_maps_r[ind, :, :]
#     fixed_p, p_reject = p_adj_maps(res_maps_p[ind, :, 0:171], mask=mask_fb, method='fdr_bh')
#     #temp_data[fixed_p > 0.05] = 0
#     temp_data_2[res_maps_p[ind, :, :] > 0.05] = 0
#     masked_data_2 = np.ma.masked_where(mask_fb != 1, temp_data_2)
#     im2 = axs2[v].imshow(masked_data_2, cmap='coolwarm', vmin=-0.8, vmax=0.8)
#     axs2[v].set_title(stim_names_pain[sense])
#     axs2[v].axis('off')
#
# fig2.colorbar(im2)
# fig2.suptitle('correlation between maps and ' + corr_with_this, size=20, va='top')
# plt.savefig(figloc+'correlation_pain_vs_' + corr_with_this + '_no_fdr.png')
# plt.close()
#
# # SENSITIVITY
#
#
# fig4, axs4 = plt.subplots(1, 3, figsize=(15, 6), facecolor='w', sharex=True, sharey=True)
# fig4.subplots_adjust(hspace= 0 , wspace= 0 )
#
# axs4 = axs4.ravel()
#
# for v, sense in enumerate(stim_names_sensitivity.keys()):
#     ind = order_maps.index(sense)
#     temp_data_4 = res_maps_r[ind, :, :]
#     fixed_p, p_reject = p_adj_maps(res_maps_p[ind, :, 0:171], mask=mask_fb, method='fdr_bh')
#     #temp_data[fixed_p > 0.05] = 0
#     temp_data_4[res_maps_p[ind, :, :] > 0.05] = 0
#     masked_data_4 = np.ma.masked_where(mask_fb != 1, temp_data_4)
#     im4 = axs4[v].imshow(masked_data_4, cmap='coolwarm', vmin=-0.8, vmax=0.8)
#     axs4[v].set_title(stim_names_sensitivity[sense])
#     axs4[v].axis('off')
#
# fig4.colorbar(im4)
# fig4.suptitle('correlation between maps and ' + corr_with_this, size=20, va='top')
# plt.savefig(figloc+'correlation_sensitivity_vs_' + corr_with_this + '_no_fdr.png')
# plt.close()

# EMOTIONS


fig6, axs6 = plt.subplots(1, 7, figsize=(15, 6), facecolor='w', sharex=True, sharey=True)
fig6.subplots_adjust(hspace=0, wspace=0)

axs6 = axs6.ravel()

for v, emotion in enumerate(stim_names_emotions.keys()):
    print(emotion)
    ind = order_maps.index(emotion)
    temp_data_6 = res_maps_r[ind, :, 0:171]
    temp_data_6[np.isnan(temp_data_6)] = 0
    fixed_p, p_reject = p_adj_maps(res_maps_p[ind, :, 0:171], mask=mask_one, method='fdr_bh')
    #temp_data[fixed_p > 0.05] = 0
    temp_data_6[res_maps_p[ind, :, 0:171] > 0.05] = 0
    masked_data_6 = np.ma.masked_where(mask_one != 1, temp_data_6)
    im6 = axs6[v].imshow(masked_data_6, cmap='coolwarm', vmin=-0.8, vmax=0.8)
    axs6[v].set_title(stim_names_emotions[emotion])
    axs6[v].axis('off')

fig6.colorbar(im6)
fig6.suptitle('correlation between maps and ' + corr_with_this, size=20, va='top')
plt.savefig(figloc+'correlation_emotions_vs_' + corr_with_this + '_no_fdr.png')
plt.close()


if also_bpi:
    fig1, axs1 = plt.subplots(1, 2, figsize=(15, 6), facecolor='w', sharex=True, sharey=True)
    fig1.subplots_adjust(hspace= 0 , wspace= 0 )

    axs1 = axs1.ravel()

    for v, sense in enumerate(stim_names_pain.keys()):
        ind = order_maps.index(sense)
        temp_data = res_maps_r_bpi[ind, :, :]
        fixed_p, p_reject = p_adj_maps(res_maps_p_bpi[ind, :, 0:171], mask=mask_fb, method='fdr_bh')
        #temp_data[fixed_p > 0.05] = 0
        temp_data[res_maps_p_bpi[ind, :, :] > 0.05] = 0
        masked_data = np.ma.masked_where(mask_fb != 1, temp_data)
        im1 = axs1[v].imshow(masked_data, cmap='coolwarm', vmin=-0.8, vmax=0.8)
        axs1[v].set_title(stim_names_pain[sense])
        axs1[v].axis('off')

    fig1.colorbar(im1)
    fig1.suptitle('correlation between maps and BPI pain', size=20, va='top')
    plt.savefig(figloc+'correlation_pain_vs_bpi_no_fdr.png')
    plt.close()

    fig3, axs3 = plt.subplots(1, 3, figsize=(15, 6), facecolor='w', sharex=True, sharey=True)
    fig3.subplots_adjust(hspace= 0 , wspace= 0 )

    axs3 = axs3.ravel()

    for v, sense in enumerate(stim_names_sensitivity.keys()):
        ind = order_maps.index(sense)
        temp_data_3 = res_maps_r_bpi[ind, :, :]
        fixed_p, p_reject = p_adj_maps(res_maps_p_bpi[ind, :, 0:171], mask=mask_fb, method='fdr_bh')
        #temp_data[fixed_p > 0.05] = 0
        temp_data_3[res_maps_p_bpi[ind, :, :] > 0.05] = 0
        masked_data_3 = np.ma.masked_where(mask_fb != 1, temp_data_3)
        im3 = axs3[v].imshow(masked_data_3, cmap='coolwarm', vmin=-0.8, vmax=0.8)
        axs3[v].set_title(stim_names_sensitivity[sense])
        axs3[v].axis('off')

    fig3.colorbar(im3)
    fig3.suptitle('correlation between maps and BPI pain', size=20, va='top')
    plt.savefig(figloc+'correlation_sensitivity_vs_bpi_no_fdr.png')
    plt.close()

    fig5, axs5 = plt.subplots(1, 7, figsize=(15, 6), facecolor='w', sharex=True, sharey=True)
    fig5.subplots_adjust(hspace= 0 , wspace= 0 )

    axs5 = axs5.ravel()

    for v, emotion in enumerate(stim_names_emotions.keys()):
        ind = order_maps.index(emotion)
        temp_data_5 = res_maps_r_bpi[ind, :, 0:171]
        temp_data_5[np.isnan(temp_data_5)] = 0
        fixed_p, p_reject = p_adj_maps(res_maps_p_bpi[ind, :, 0:171], mask=mask_one, method='fdr_bh')
        #temp_data[fixed_p > 0.05] = 0
        temp_data_5[res_maps_p_bpi[ind, :, 0:171] > 0.05] = 0
        masked_data_5 = np.ma.masked_where(mask_one != 1, temp_data_5)
        im5 = axs5[v].imshow(masked_data_5, cmap='coolwarm', vmin=-0.8, vmax=0.8)
        axs5[v].set_title(stim_names_emotions[emotion])
        axs5[v].axis('off')

    fig5.colorbar(im5)
    fig5.suptitle('correlation between maps and BPI pain', size=20, va='top')
    plt.savefig(figloc+'correlation_emotions_vs_bpi_no_fdr.png')
    plt.close()

# corr_map_spearman[p_map_spearman > 0.05] = 0
# showdata = corr_map_spearman
#
# fig = plt.figure()
# if onesided:
#     img = plt.imshow(showdata, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
# else:
#     img = plt.imshow(showdata, cmap=twosided_cmap, vmin=0, vmax=1)
#
# fig.colorbar(img)
# plt.savefig(figloc+cond+'_spearman_corr_with_'+corr_with_this+'_p_lim.png')
# plt.close()

