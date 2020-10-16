import matplotlib
# matplotlib.use('Agg')

from classdefinitions import Subject, Stimuli
from bodyfunctions import *
import h5py
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

figloc = '/m/nbe/scratch/socbrain/kipupotilaat/figures/'
maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'
dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/stockholm/processed/'
datafile = get_latest_datafile(dataloc)

mask_fb = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')
mask_one = read_in_mask(maskloc + 'mask_front_new.png')

stim_names = {'emotions_0': ['sadness', 0], 'emotions_1': ['happiness', 0], 'emotions_2': ['anger', 0],
              'emotions_3': ['surprise', 0], 'emotions_4': ['fear', 0], 'emotions_5': ['disgust', 0],
              'emotions_6': ['neutral', 0],
              'pain_0': ['acute pain', 1], 'pain_1': ['chonic pain', 1], 'sensitivity_0': ['tactile sensitivity', 1],
              'sensitivity_1': ['nociceptive sensitivity', 1], 'sensitivity_2': ['hedonic sensitivity', 1]}

curr_fig = 'emotions_1'

with h5py.File(datafile, 'r') as kipu:
    fig = kipu[curr_fig].value
    #group = kipu['groups'].value


if stim_names[curr_fig][1] == 0:
    mask = mask_one
else:
    mask = mask_fb

#group_1 = fig[group == 'LOWER_BACK',:,:]
#group_0 = fig[group == 'FIBROMYALGI',:,:]
#
# g_kipu = np.arange(0, 102)
# g_control = np.arange(102, 204)
# stats_1, pvals_1 = compare_groups(group_1, group_0)
# pvals_1[np.isnan(pvals_1)]=1
# print("calculating t-test")
# statistics_twosamp, pval_twosamp = stats.ttest_ind(group_1, group_0, axis=0, nan_policy='omit')
# print("t-test ready")
# ##
# fig = plt.figure()
# show_now = stats_1
# masked_data = np.ma.masked_where(mask != 1, show_now)
# masked_thresholded_data = masked_data[pvals_1 < 0.05]
# img = plt.imshow(masked_data, cmap='seismic')
# # #
# fig.colorbar(img)
# # # # plt.show()
# # plt.show()
# plt.savefig(figloc+'temp_pval.png')
# plt.close()

#
# for i, cond in enumerate(stim_names.keys()):
#
#     with h5py.File(datafile, 'r') as p:
#         pain = p[cond].value
#     with h5py.File(datafile_controls, 'r') as c:
#         control = c[cond].value
#
#     if stim_names[cond][1] == 1:
#         mask = mask_fb
#         cmap = 'hot'
#         vmin= 0
#         fig = plt.figure(figsize=(25, 10))
#
#     else:
#         mask = mask_one
#         cmap = 'coolwarm'
#         vmin = -10
#         fig= plt.figure(figsize=(10,10))
#
#     control_t, control_p = stats.ttest_1samp(control, 0, nan_policy='omit', axis=0)
#     control_p_corrected = p_adj_maps(control_p, mask)
#     control_p_corrected[np.isnan(control_p_corrected)] = 1
#     control_t[control_p_corrected>0.05] = 0
#     masked_control= np.ma.masked_where(mask != 1,control_t)
#
#     kipu_t, kipu_p = stats.ttest_1samp(pain, 0, nan_policy='omit', axis=0)
#     kipu_p_corrected = p_adj_maps(kipu_p, mask)
#     kipu_p_corrected[np.isnan(kipu_p_corrected)] = 1
#     kipu_t[kipu_p_corrected>0.05] = 0
#     masked_kipu= np.ma.masked_where(mask != 1,kipu_t)
#
#     # try also paired two sample t-test?
#     twosamp_t, twosamp_p = stats.ttest_ind(control,pain, axis=0, nan_policy='omit')
#     twosamp_p_corrected = p_adj_maps(twosamp_p, mask)
#     twosamp_p_corrected[np.isnan(twosamp_p_corrected)] = 1
#     twosamp_t[twosamp_p_corrected>0.05] =0
#     masked_twosamp = np.ma.masked_where(mask != 1,twosamp_t)
#
#     ax1 = plt.subplot(132)
#     img1 = plt.imshow(masked_kipu, cmap=cmap, vmin=vmin, vmax=10)
#     ax1.title.set_text('All pain')
#     fig.colorbar(img1,fraction=0.046, pad=0.04)
#     ax1.axis('off')
#
#     ax2 = plt.subplot(131)
#     img2 = plt.imshow(masked_control, cmap=cmap, vmin=vmin, vmax=10)
#     ax2.title.set_text('Controls')
#     fig.colorbar(img2, fraction=0.046, pad=0.04)
#     ax2.axis('off')
#
#     ax3 = plt.subplot(133)
#     img3 = plt.imshow(masked_twosamp, cmap='bwr', vmin=-10, vmax=10)
#     ax3.title.set_text('Difference')
#     fig.colorbar(img3, fraction=0.046, pad=0.04)
#     ax3.axis('off')
#     #
#     fig.suptitle(stim_names[cond][0], size=20, va='top')
#     #plt.show()
#     plt.savefig(figloc+cond+'.png')
#     #plt.close()

res_crps_pix = []
res_crps_prop = []
crps_conds = []
res_kipu_pix = []
res_kipu_prop = []
kipu_conds = []

for i, cond in enumerate(stim_names.keys()):

    with h5py.File(datafile, 'r') as p:
        pain = p[cond].value
    with h5py.File(datafile_controls, 'r') as c:
        control = c[cond].value

    if stim_names[cond][1]:
        mask_use = mask_fb
    else:
        mask_use = mask_one

    pix_crps, prop_crps = count_pixels(pain, mask=mask_use)
    res_crps_pix.extend(pix_crps)
    res_crps_prop.extend(prop_crps)
    crps_conds.extend(np.repeat(stim_names[cond][0], len(pix_crps)))

    pix_kipu, prop_kipu = count_pixels(control, mask=mask_use)
    res_kipu_pix.extend(pix_kipu)
    res_kipu_prop.extend(prop_kipu)
    kipu_conds.extend(np.repeat(stim_names[cond][0], len(pix_kipu)))

res = pd.DataFrame({'condition': np.append(crps_conds, kipu_conds), 'pixels': np.append(res_crps_pix, res_kipu_pix),
                    'proportion': np.append(res_crps_prop, res_kipu_prop),
                    "group": np.append(np.repeat('pain patients', len(res_crps_pix)), np.repeat('matched controls', len(res_kipu_pix)),
                                       axis=0)})

stim_names_emotions = {'emotions_0': 'sadness', 'emotions_1': 'happiness', 'emotions_2': 'anger',
                       'emotions_3': 'surprise',
                       'emotions_4': 'fear', 'emotions_5': 'disgust', 'emotions_6': 'neutral'}

stim_names_pain = {'pain_0': 'acute pain', 'pain_1': 'chonic pain'}
stim_names_sensitivity = {'sensitivity_0': 'tactile sensitivity',
                          'sensitivity_1': 'nociceptive sensitivity', 'sensitivity_2': 'hedonic sensitivity'}

visualise = res[res.condition.isin(stim_names_emotions.values())]
fig = plt.figure()
ax = sns.swarmplot(data=visualise, x='condition', y='proportion', hue='group', dodge=True, color=".4", size=2.5)
ax = sns.boxplot(data=visualise, x='condition', y='proportion', hue='group', showfliers=False, notch=True)
# plt.show()
plt.savefig(figloc + 'emotion_proportion_coloured_kipu_controls.png')
plt.close()

# stats.ttest_ind(res[(res.condition == 'surprise')&(res.group == 'pain patients')]['pixels'],res[(res.condition == 'surprise') &(res.group =='matched controls')]['pixels'])
# stats.ttest_ind(res[(res.condition == 'fear')&(res.group == 'pain patients')]['pixels'],res[(res.condition == 'fear') &(res.group =='matched controls')]['pixels'])
# stats.ttest_ind(res[(res.condition == 'disgust')&(res.group == 'pain patients')]['pixels'],res[(res.condition == 'disgust') &(res.group =='matched controls')]['pixels'])
# stats.ttest_ind(res[(res.condition == 'sadness')&(res.group == 'pain patients')]['pixels'],res[(res.condition == 'sadness') &(res.group =='matched controls')]['pixels'])
# stats.ttest_ind(res[(res.condition == 'happiness')&(res.group == 'pain patients')]['pixels'],res[(res.condition == 'happiness') &(res.group =='matched controls')]['pixels'])
# stats.ttest_ind(res[(res.condition == 'anger')&(res.group == 'pain patients')]['pixels'],res[(res.condition == 'anger') &(res.group =='matched controls')]['pixels'])
# stats.ttest_ind(res[(res.condition == 'neutral')&(res.group == 'pain patients')]['pixels'],res[(res.condition == 'neutral') &(res.group =='matched controls')]['pixels'])
#
# stats.ttest_ind(res[(res.condition == 'hedonic sensitivity')&(res.group == 'pain patients')]['pixels'],res[(res.condition == 'hedonic sensitivity') &(res.group =='matched controls')]['pixels'])
# stats.ttest_ind(res[(res.condition == 'nociceptive sensitivity')&(res.group == 'pain patients')]['pixels'],res[(res.condition == 'nociceptive sensitivity') &(res.group =='matched controls')]['pixels'])
# stats.ttest_ind(res[(res.condition == 'tactile sensitivity')&(res.group == 'pain patients')]['pixels'],res[(res.condition == 'tactile sensitivity') &(res.group =='matched controls')]['pixels'])
#
# stats.ttest_ind(res[(res.condition == 'acute pain')&(res.group == 'pain patients')]['pixels'],res[(res.condition == 'acute pain') &(res.group =='matched controls')]['pixels'])
# stats.ttest_ind(res[(res.condition == 'chonic pain')&(res.group == 'pain patients')]['pixels'],res[(res.condition == 'chonic pain') &(res.group =='matched controls')]['pixels'])
#


# control['bg'][['age','work_sitting','work_physical', 'sex']].astype(float).describe(include='all')
# sum(control['bg']['sex'].astype(int))
# kipu['bg'][['age','work_sitting','work_physical', 'sex']].astype(float).describe(include='all')
# sum(control['bg']['sex'].astype(int))
#
# control['bg'][['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_surprise','feels_disgust']].astype(int).describe(include='all')
# kipu['bg'][['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_surprise','feels_disgust']].astype(int).describe(include='all')
#
#
# kipu['bg'][['pain_now','pain_last_day','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
#                 'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns','hist_crps','hist_fibro']].astype(int).describe(include='all')
# control['bg'][['pain_now','pain_last_day','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
#                 'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns']].crosstab(columns="count")
#
# pd.crosstab(control['bg'][['pain_now','pain_last_day','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
#                 'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns']], columns="count")
