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

mask_fb = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')
mask_one = read_in_mask(maskloc + 'mask_front_new.png')

stim_names = {'emotions_0': ['sadness', 0], 'emotions_1': ['happiness', 0], 'emotions_2': ['anger', 0],
              'emotions_3': ['surprise', 0], 'emotions_4': ['fear', 0], 'emotions_5': ['disgust', 0],
              'emotions_6': ['neutral', 0],
              'pain_0': ['acute pain', 1], 'pain_1': ['chonic pain', 1], 'sensitivity_0': ['tactile sensitivity', 1],
              'sensitivity_1': ['nociceptive sensitivity', 1], 'sensitivity_2': ['hedonic sensitivity', 1]}


curr = mask_fb.copy()
right_front = curr[:, 1:85]
left_front = curr[:, 85:169]
left_back = curr[:, 172:256]
right_back = curr[:, 256:340]

# left_back.shape[1] - left_front.shape[1]
# right_back.shape[1] - right_front.shape[1]
#
# left = left_back + np.flip(left_front, axis=1)
# right = right_back + np.flip(right_front, axis=1)
#
# fig = plt.figure()
#
# ax1 = plt.subplot(121)
# img1 = plt.imshow(left)
#
# ax2 = plt.subplot(122)
# img2 = plt.imshow(right)
# plt.savefig(figloc + 'sides_combined_7.png')
# plt.close()

with h5py.File(datafile, 'r') as h:
    kipu_chronic = h['pain_1'].value
    kipu_acute =  h['pain_0'].value
    subids = h['subid'].value
    kipu_diagnoses = list(h['groups'])
    crps_indices = np.asarray([x == 'CRPS' for x in kipu_diagnoses])

crps_patients = subids[crps_indices]
side_pain = np.zeros((1,crps_patients.shape[0]))
crps_patients_with_side = np.vstack((crps_patients, side_pain))

for subnum in crps_patients:
    sub = np.where(subids == subnum)[0][0]
    sub_chronic = kipu_chronic[sub, :, :]
    sub_acute = kipu_acute[sub, :, :]
    chronic_pain_left = sum(sum(np.concatenate((sub_chronic[:, 85:169], sub_chronic[:, 172:256]))))
    chronic_pain_right = sum(sum(np.concatenate((sub_chronic[:, 1:85], sub_chronic[:, 256:340]))))
    acute_pain_left = sum(sum(np.concatenate((sub_acute[:, 85:169], sub_acute[:, 172:256]))))
    acute_pain_right = sum(sum(np.concatenate((sub_acute[:, 1:85], sub_acute[:, 256:340]))))
    print(str(subnum), ' Chronic: ', str(chronic_pain_left-chronic_pain_right), ' acute: ', str(acute_pain_left-acute_pain_right))
    crps_patients_with_side[1,np.where(crps_patients == subnum)[0][0]] = chronic_pain_left-chronic_pain_right

res_maps_t = np.zeros((len(stim_names.keys()), 522, 168))
res_maps_p = np.zeros((len(stim_names.keys()), 522, 168))
order_maps = []

for j, pic in enumerate(stim_names.keys()):
    order_maps.append(pic)
    with h5py.File(datafile, 'r') as h:
        data = h[pic].value
    if stim_names[pic][1] == 1:
        pain_side = np.zeros((crps_patients.shape[0], 522, 168))
        nonpain_side = np.zeros((crps_patients.shape[0], 522, 168))
    else:
        pain_side = np.zeros((crps_patients.shape[0], 522, 84))
        nonpain_side = np.zeros((crps_patients.shape[0], 522, 84))


    for subnum in crps_patients:
        sub = np.where(subids == subnum)[0][0]
        subloc = np.where(crps_patients_with_side[0] == subnum)[0][0]
        # align all sides to right front & back
        # if left side is pain side, rotate pain, do not rotate non-pain
        if crps_patients_with_side[1, subloc] > 0:
            pain_side[subloc, :, 0:84] = np.flip(data[sub, :, 85:169], axis=1)
            nonpain_side[subloc, :, 0:84] = data[sub, :, 1:85]
            if stim_names[pic][1] == 1:
                pain_side[subloc, :, 84:168] = np.flip(data[sub, :, 172:256], axis=1)
                nonpain_side[subloc, :, 84:168] = data[sub, :, 256:340]
        # if right side is pain side, do not rotate pain, rotate non-pain
        elif crps_patients_with_side[1, subloc] < 0:
            pain_side[subloc, :, 0:84] = data[sub, :, 1:85]
            nonpain_side[subloc, :, 0:84] = np.flip(data[sub, :, 85:169], axis=1)
            if stim_names[pic][1] == 1:
                pain_side[subloc, :, 84:168] = data[sub, :, 256:340]
                nonpain_side[subloc, :, 84:168] = np.flip(data[sub, :, 172:256], axis=1)
    if stim_names[pic][1] == 1:
        res_maps_t[j, :, :], res_maps_p[j, :, :] = stats.ttest_rel(pain_side, nonpain_side, axis=0)
    else:
        res_maps_t[j, :, 0:84], res_maps_p[j, :, 0:84] = stats.ttest_rel(pain_side, nonpain_side, axis=0)


stim_names_emotions = {'emotions_0':'sadness', 'emotions_1':'happiness', 'emotions_2':'anger', 'emotions_3':'surprise',
              'emotions_4': 'fear', 'emotions_5':'disgust', 'emotions_6':'neutral'}

mask_half = mask_one[:, 1:85]


fig, axs = plt.subplots(1,7, figsize=(15, 6), facecolor='w', sharex=True, sharey=True)
fig.subplots_adjust(hspace= 0 , wspace= 0 )

axs = axs.ravel()

for b, emotion in enumerate(stim_names_emotions.keys()):
    ind = order_maps.index(emotion)
    fixed_p, p_reject = p_adj_maps(res_maps_p[ind, :, 0:84], mask=mask_half, method='fdr_bh')
    temp_data = res_maps_t[ind, :, 0:84]
    #temp_data[fixed_p > 0.05] = 0
    temp_data[res_maps_p[ind, :, 0:84] > 0.05] = 0
    masked_data = np.ma.masked_where(mask_half != 1, temp_data)
    im = axs[b].imshow(masked_data, cmap='coolwarm', vmin=-10, vmax=10)
    axs[b].set_title(stim_names_emotions[emotion])
    axs[b].axis('off')

fig.colorbar(im)
fig.suptitle('CRPS patients: pain vs nonpain side', size=20, va='top')
plt.savefig(figloc+'emotions_crps_pain_vs_nopain_no_fdr.png')
plt.close()


stim_names_sensitivity = {'sensitivity_0':'tactile sensitivity',
                          'sensitivity_1':'nociceptive sensitivity', 'sensitivity_2':'hedonic sensitivity'}

mask_one_space = np.hstack((mask_one[:, 1:85], np.zeros((522, 10)), mask_one[:, 85:169]))

fig1, axs1 = plt.subplots(1, 3, figsize=(15, 6), facecolor='w', sharex=True, sharey=True)
fig1.subplots_adjust(hspace= 0 , wspace= 0 )

axs1 = axs1.ravel()

for v, sense in enumerate(stim_names_sensitivity.keys()):
    ind = order_maps.index(sense)
    fixed_p, p_reject = p_adj_maps(res_maps_p[ind, :, :], mask=mask_one[:, 1:169], method='fdr_bh')
    temp_data = res_maps_t[ind, :, :]
    # temp_data[fixed_p > 0.05] = 0
    temp_data[res_maps_p[ind, :, :] > 0.05] = 0
    temp_data_spaced_out = np.hstack((temp_data[:,0:84], np.zeros((522, 10)), temp_data[:,84:168]))
    masked_data = np.ma.masked_where(mask_one_space != 1, temp_data_spaced_out)
    im1 = axs1[v].imshow(masked_data, cmap='coolwarm', vmin=-5, vmax=5)
    axs1[v].set_title(stim_names_sensitivity[sense])
    axs1[v].axis('off')
    axs1[v].text(1, 80, 'front')
    axs1[v].text(140, 80, 'back')


fig.colorbar(im1)
fig1.suptitle('CRPS patients: pain vs nonpain side', size=20, va='top')
plt.savefig(figloc+'sensitivity_crps_pain_vs_nopain_no_fdr.png')
plt.close()


stim_names_pain = {'pain_0':'acute pain', 'pain_1': 'chonic_pain'}

fig2, axs2 = plt.subplots(1, 2, figsize=(15, 6), facecolor='w', sharex=True, sharey=True)
fig2.subplots_adjust(hspace= 0 , wspace= 0 )

axs2 = axs2.ravel()

for k, pain in enumerate(stim_names_pain.keys()):
    ind = order_maps.index(pain)
    fixed_p, p_reject = p_adj_maps(res_maps_p[ind, :, :], mask=mask_one[:, 1:169], method='fdr_bh')
    temp_data = res_maps_t[ind, :, :]
    temp_data[fixed_p > 0.05] = 0
    #temp_data[res_maps_p[ind, :, :] > 0.05] = 0
    temp_data_spaced_out = np.hstack((temp_data[:,0:84], np.zeros((522, 10)), temp_data[:,84:168]))
    masked_data = np.ma.masked_where(mask_one_space != 1, temp_data_spaced_out)
    im1 = axs2[k].imshow(masked_data, cmap='coolwarm', vmin=-5, vmax=5)
    axs2[k].set_title(stim_names_pain[pain])
    axs2[k].axis('off')
    axs2[k].text(1, 80, 'front')
    axs2[k].text(140, 80, 'back')


fig2.colorbar(im1)
fig2.suptitle('CRPS patients: pain vs nonpain side', size=20, va='top')
plt.savefig(figloc+'pain_crps_pain_vs_nopain_fdr_bh.png')
plt.close()
