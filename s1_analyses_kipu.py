import matplotlib
#matplotlib.use('Agg')

from classdefinitions import Subject, Stimuli
from bodyfunctions import *
import pickle
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

figloc = '/Users/jtsuvile/Documents/projects/kipupotilaat/data/figures/'
maskloc = '/Volumes/SCRsocbrain/kipupotilaat/data/'
datafile = '/Users/jtsuvile/Documents/projects/kipupotilaat/data/full_dataset.pickle'
datafile_controls = '/Users/jtsuvile/Documents/projects/kipupotilaat/data/matched_controls.pickle'
print("loading data")
kipu = pd.read_pickle(datafile)
control = pd.read_pickle(datafile_controls)
print("loaded data")
mask_fb = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')
mask_one = read_in_mask(maskloc + 'mask_front_new.png')

stim_names = {'emotions_0':'sadness', 'emotions_1':'happiness', 'emotions_2':'anger', 'emotions_3':'surprise',
              'emotions_4': 'fear', 'emotions_5':'disgust', 'emotions_6':'neutral',
              'pain_0':'acute pain', 'pain_1': 'chonic_pain', 'sensitivity_0':'tactile sensitivity',
              'sensitivity_1':'nociceptive sensitivity', 'sensitivity_2':'hedonic sensitivity'}

twosided = [0,0,0,0,0,0,0,1,1,1,1,1]

#data_to_function = np.concatenate((kipu['emotions_1'], control['emotions_1']), axis=0)
#g_kipu = np.arange(0, 102)
#g_control = np.arange(102, 204)
#stats, pvals = compare_groups(data_to_function, g_kipu, g_control)
# print("calculating t-test")
# statistics_twosamp, pval_twosamp = stats.ttest_ind(kipu['emotions_1'], control['emotions_1'], axis=0, nan_policy='omit')
# print("t-test ready")
# ##
# fig = plt.figure()
#
# show_now = np.nanmean(control['pain_0'],axis=0)
# masked_data = np.ma.masked_where(mask_fb != 1, show_now)
# # masked_thresholded_data = masked_data[pval_twosamp < 0.05]
# img = plt.imshow(masked_data, cmap='seismic')
# #
# fig.colorbar(img)
# # # plt.show()
# plt.show()
# #plt.savefig(figloc+'temp_pval.png')
# #plt.close()


# for i, cond in enumerate(stim_names.keys()):
#
#     if twosided[i]:
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
#     control_t, control_p = stats.ttest_1samp(control[cond], 0, nan_policy='omit', axis=0)
#     control_p[np.isnan(control_p)] = 1
#     control_t[control_p>0.05] = 0
#     masked_control= np.ma.masked_where(mask != 1,control_t)
#
#     kipu_t, kipu_p = stats.ttest_1samp(kipu[cond], 0, nan_policy='omit', axis=0)
#     kipu_p[np.isnan(kipu_p)] = 1
#     kipu_t[kipu_p>0.05] = 0
#     masked_kipu= np.ma.masked_where(mask != 1,kipu_t)
#
#     # try also paired two sample t-test?
#     twosamp_t, twosamp_p = stats.ttest_ind(control[cond],kipu[cond], axis=0, nan_policy='omit')
#     twosamp_p[np.isnan(twosamp_p)] = 1
#     twosamp_t[twosamp_p>0.05] =0
#     masked_twosamp = np.ma.masked_where(mask != 1,twosamp_t)
#
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
#     fig.suptitle(stim_names[cond], size=20, va='top')
#     #plt.show()
#     plt.savefig(figloc+cond+'.png')
#     #plt.close()

# res_crps_pix = []
# res_crps_prop = []
# crps_conds = []
# res_kipu_pix = []
# res_kipu_prop = []
# kipu_conds = []
# for i, cond in enumerate(stim_names.keys()):
#     if twosided[i]:
#         mask_use = mask_fb
#     else:
#         mask_use = mask_one
#     pix_crps, prop_crps = count_pixels(kipu[cond], mask=mask_use)
#     res_crps_pix.extend(pix_crps)
#     res_crps_prop.extend(prop_crps)
#     crps_conds.extend(np.repeat(stim_names[cond], len(pix_crps)))
#
#     pix_kipu, prop_kipu = count_pixels(control[cond], mask=mask_use)
#     res_kipu_pix.extend(pix_kipu)
#     res_kipu_prop.extend(prop_kipu)
#     kipu_conds.extend(np.repeat(stim_names[cond], len(pix_kipu)))
#
# res = pd.DataFrame({'condition': np.append(crps_conds, kipu_conds), 'pixels': np.append(res_crps_pix, res_kipu_pix),
#                     'proportion': np.append(res_crps_prop, res_kipu_prop),
#                     "group": np.append(np.repeat('pain patients', len(res_crps_pix)), np.repeat('matched controls', len(res_kipu_pix)),
#                                        axis=0)})
#
# stim_names_emotions = {'emotions_0': 'sadness', 'emotions_1': 'happiness', 'emotions_2': 'anger',
#                        'emotions_3': 'surprise',
#                        'emotions_4': 'fear', 'emotions_5': 'disgust', 'emotions_6': 'neutral'}
#
# stim_names_pain = {'pain_0': 'acute pain', 'pain_1': 'chonic_pain'}
# stim_names_sensitivity = {'sensitivity_0': 'tactile sensitivity',
#                           'sensitivity_1': 'nociceptive sensitivity', 'sensitivity_2': 'hedonic sensitivity'}
#
# visualise = res[res.condition.isin(stim_names_emotions.values())]
# fig = plt.figure()
# ax = sns.swarmplot(data=visualise, x='condition', y='proportion', hue='group', dodge=True, color=".4", size=2.5)
# ax = sns.boxplot(data=visualise, x='condition', y='proportion', hue='group', showfliers=False, notch=True)
# # plt.show()
# plt.savefig(figloc + 'emotion_proportion_coloured_kipu_controls.png')
# plt.close()

control['bg'][['age','work_sitting','work_physical', 'sex']].astype(float).describe(include='all')
sum(control['bg']['sex'].astype(int))
kipu['bg'][['age','work_sitting','work_physical', 'sex']].astype(float).describe(include='all')
sum(control['bg']['sex'].astype(int))

control['bg'][['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_surprise','feels_disgust']].astype(int).describe(include='all')
kipu['bg'][['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_surprise','feels_disgust']].astype(int).describe(include='all')


kipu['bg'][['pain_now','pain_last_day','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns','hist_crps','hist_fibro']].astype(int).describe(include='all')
control['bg'][['pain_now','pain_last_day','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns']].crosstab(columns="count")

pd.crosstab(control['bg'][['pain_now','pain_last_day','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns']], columns="count")
