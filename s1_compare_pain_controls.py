from classdefinitions import Subject, Stimuli
from bodyfunctions import *
import h5py
import numpy as np
import pandas as pd
from scipy import stats
import csv
import matplotlib.pyplot as plt
import seaborn as sns


figloc = '/m/nbe/scratch/socbrain/kipupotilaat/figures/'
maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'
dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/'
datafile = get_latest_datafile(dataloc)

dataloc_controls = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/matched_controls/'
datafile_controls = get_latest_datafile(dataloc_controls)

mask_fb = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')
mask_one = read_in_mask(maskloc + 'mask_front_new.png')

stim_names = {'emotions_0': ['sadness', 0], 'emotions_1': ['happiness', 0], 'emotions_2': ['anger', 0],
              'emotions_3': ['surprise', 0], 'emotions_4': ['fear', 0], 'emotions_5': ['disgust', 0],
              'emotions_6': ['neutral', 0],
              'pain_0': ['acute pain', 1], 'pain_1': ['chonic_pain', 1], 'sensitivity_0': ['tactile sensitivity', 1],
              'sensitivity_1': ['nociceptive sensitivity', 1], 'sensitivity_2': ['hedonic sensitivity', 1]}


# Visualise group differences

for i, cond in enumerate(stim_names.keys()):

    with h5py.File(datafile, 'r') as h:
        kipu = h[cond].value

    with h5py.File(datafile_controls, 'r') as c:
        control = c[cond].value

    if stim_names[cond][1] == 1:
        mask = mask_fb
        cmap = 'hot'
        vmin = 0
        vmax = 10
        fig = plt.figure(figsize=(25, 10))
    else:
        mask = mask_one
        cmap = 'coolwarm'
        vmin = -10
        vmax = 10
        fig= plt.figure(figsize=(14,10))

    control_t, control_p = stats.ttest_1samp(control, 0, nan_policy='omit', axis=0) #np.nanmean(binarize(control), axis=0)
    control_p_corrected, control_reject = p_adj_maps(control_p, mask, method='fdr_bh')
    #control_p_corrected = control_p
    control_p_corrected[np.isnan(control_p_corrected)] = 1
    control_t[control_p_corrected>0.05] = 0
    #control_t = np.nanmean(binarize_posneg(control),axis=0)
    masked_control= np.ma.masked_where(mask != 1,control_t)

    kipu_t, kipu_p = stats.ttest_1samp(kipu, 0, nan_policy='omit', axis=0)#np.nanmean(binarize(pain), axis=0)

    kipu_p_corrected, kipu_reject = p_adj_maps(kipu_p, mask, method='fdr_bh')
    #kipu_p_corrected = kipu_p
    kipu_p_corrected[np.isnan(kipu_p_corrected)] = 1
    kipu_t[kipu_p_corrected>0.05] = 0
    #kipu_t = np.nanmean(binarize_posneg(kipu),axis=0)
    masked_kipu= np.ma.masked_where(mask != 1,kipu_t)

    if (np.nanmin(control)==0) & (np.nanmin(kipu)==0):
        twosamp_t, twosamp_p = compare_groups(kipu, control, testtype='z')
    else:
        twosamp_t, twosamp_p = stats.ttest_ind(kipu, control, axis=0, nan_policy='omit')

    twosamp_p_corrected, twosamp_reject = p_adj_maps(twosamp_p, mask=mask, method='fdr_bh')
    #twosamp_p_corrected = twosamp_p
    twosamp_p_corrected[np.isnan(twosamp_p_corrected)] = 1
    twosamp_t_no_fdr = twosamp_t.copy()

    twosamp_t[twosamp_p_corrected > 0.05] = 0
    masked_twosamp = np.ma.masked_where(mask != 1, twosamp_t)

    twosamp_t_no_fdr[twosamp_p > 0.05] = 0
    masked_twosamp_no_fdr = np.ma.masked_where(mask != 1, twosamp_t_no_fdr)

    ax1 = plt.subplot(142)
    img1 = plt.imshow(masked_kipu, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.title.set_text('pain patients')
    fig.colorbar(img1,fraction=0.046, pad=0.04)
    ax1.axis('off')

    ax2 = plt.subplot(141)
    img2 = plt.imshow(masked_control, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.title.set_text('control patients')
    fig.colorbar(img2, fraction=0.046, pad=0.04)
    ax2.axis('off')

    ax3 = plt.subplot(143)
    img3 = plt.imshow(masked_twosamp, cmap='bwr', vmin=-8, vmax=8)
    ax3.title.set_text('Difference')
    fig.colorbar(img3, fraction=0.046, pad=0.04)
    ax3.axis('off')

    ax4 = plt.subplot(144)
    img4 = plt.imshow(masked_twosamp_no_fdr, cmap='bwr', vmin=-8, vmax=8)
    ax4.title.set_text('Difference, no FDR')
    fig.colorbar(img4, fraction=0.046, pad=0.04)
    ax4.axis('off')

    #
    fig.suptitle(stim_names[cond][0], size=20, va='top')
    #plt.show()
    plt.savefig(figloc+cond+'_controls_pain_pixelwise.png')
    plt.close()

#
#
#
# res_crps_pix = []
# res_crps_prop = []
# crps_conds = []
# res_kipu_pix = []
# res_kipu_prop = []
# kipu_conds = []
# for i, cond in enumerate(stim_names.keys()):
#     with h5py.File(datafile, 'r') as h:
#         kipu = h[cond].value
#         kipu_diagnoses = list(h['groups'])
#
#     crps_indices = np.asarray([x == 'CRPS' for x in kipu_diagnoses])
#     pain = kipu[~crps_indices]
#     crps = kipu[crps_indices]
#
#     if stim_names[cond][1] == 1:
#         mask_use = mask_fb
#     else:
#         mask_use = mask_one
#     pix_crps, prop_crps = count_pixels(crps, mask=mask_use)
#     res_crps_pix.extend(pix_crps)
#     res_crps_prop.extend(prop_crps)
#     crps_conds.extend(np.repeat(stim_names[cond][0], len(pix_crps)))
#
#     pix_kipu, prop_kipu = count_pixels(pain, mask=mask_use)
#     res_kipu_pix.extend(pix_kipu)
#     res_kipu_prop.extend(prop_kipu)
#     kipu_conds.extend(np.repeat(stim_names[cond][0], len(pix_kipu)))
#
#
# res = pd.DataFrame({'condition':np.append(crps_conds, kipu_conds), 'pixels':np.append(res_crps_pix, res_kipu_pix), 'proportion':np.append(res_crps_prop, res_kipu_prop), "group":np.append(np.repeat('crps', len(res_crps_pix)), np.repeat('kipu', len(res_kipu_pix)), axis=0)})
#
#
# stim_names_emotions = {'emotions_0':'sadness', 'emotions_1':'happiness', 'emotions_2':'anger', 'emotions_3':'surprise',
#               'emotions_4': 'fear', 'emotions_5':'disgust', 'emotions_6':'neutral'}
#
# stim_names_pain = {'pain_0':'acute pain', 'pain_1': 'chonic_pain'}
# stim_names_sensitivity = {'sensitivity_0':'tactile sensitivity',
#               'sensitivity_1':'nociceptive sensitivity', 'sensitivity_2':'hedonic sensitivity'}
#
# visualise = res[res.condition.isin(stim_names_emotions.values())]
# fig = plt.figure()
# ax = sns.swarmplot(data=visualise, x='condition', y = 'proportion', hue='group', dodge=True, color=".4", size=5)
# ax = sns.boxplot(data=visualise, x='condition', y = 'proportion', hue='group', showfliers=False, notch=True)
# #plt.show()
# plt.savefig(figloc+'emotions_proportion_coloured_kipu_crps.png')
# plt.close()
# #
# kipu['bg'][crps][['age','work_sitting','work_physical', 'sex']].astype(float).describe(include='all')
# sum(kipu['bg'][crps]['sex'].astype(int))
# kipu['bg'][~crps][['age','work_sitting','work_physical', 'sex']].astype(float).describe(include='all')
# sum(kipu['bg'][~crps]['sex'].astype(int))
#
# kipu['bg'][crps][['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry',
#                   'feels_surprise','feels_disgust']].astype(int).describe(include='all')
# kipu['bg'][~crps][['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry',
#                    'feels_surprise','feels_disgust']].astype(int).describe(include='all')
#
#
# kipu['bg'][crps][['pain_now','pain_last_day','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
#                 'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns','hist_crps','hist_fibro']].astype(int).describe(include='all')
# kipu['bg'][~crps][['pain_now','pain_last_day','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
#                 'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns']].crosstab(columns="count")
#
# pd.crosstab(kipu['bg'][crps]['pain_now'], columns="count")
# #'pain_last_day','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual', 'hist_crps', 'hist_fibro'
# # 'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns'
# stats.ttest_ind(res[(res.condition == 'surprise')&(res.group == 'crps')]['pixels'],res[(res.condition == 'surprise') &(res.group =='kipu')]['pixels'])