from bodyfunctions import *
import numpy as np
import pandas as pd
import random

dataloc_controls = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/'
datafile_controls = get_latest_datafile(dataloc_controls)
controls_bg = pd.DataFrame(columns=['subid','age','sex','pain_now','feels_pain','pain_chronic',
                                    'hist_abdomen', 'hist_back_shoulder', 'hist_headache', 'hist_joint_limb',
                                    'hist_menstrual', 'hist_migraine'])

with h5py.File(datafile_controls, 'r') as c:
    for column in controls_bg:
        controls_bg[column] = c[column][:]

controls_bg.set_index('subid', drop=False, inplace=True)
controls_bg = controls_bg[(controls_bg.pain_chronic == 0) & (controls_bg.pain_now == 0)].copy()

acceptable_controls_original = controls_bg.copy()
# controls_bg_more_stringent = controls_bg[((controls_bg.pain_chronic == 0) | (controls_bg.pain_chronic == 3)) &
#                                          ((controls_bg.hist_abdomen == 0) | (controls_bg.hist_abdomen == 3)) &
#                                          ((controls_bg.hist_back_shoulder == 0) | (controls_bg.hist_back_shoulder == 3)) &
#                                          ((controls_bg.hist_headache == 0) | (controls_bg.hist_headache == 3)) &
#                                          ((controls_bg.hist_joint_limb == 0) | (controls_bg.hist_joint_limb == 3)) &
#                                          ((controls_bg.hist_menstrual == 0) | (controls_bg.hist_menstrual == 3)) &
#                                          ((controls_bg.hist_migraine == 0) | (controls_bg.hist_migraine == 3))].copy()

dataloc_pain = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/'
datafile_pain = get_latest_datafile(dataloc_pain)
pain_bg = pd.DataFrame(columns=['subid','age','sex','pain_now','groups'])

with h5py.File(datafile_pain, 'r') as b:
    for column in pain_bg:
        pain_bg[column] = b[column][:]

pain_bg.set_index('subid', inplace=True, drop=False)
#
age_diff_cutoff = 3
min_sum = 1000
for i in range(1,500):
    matches = pain_bg.copy()
    matches['control_id'] = 0
    matches['control_sex'] = np.nan
    matches['control_age'] = 0
    matches['control_pain_now'] = 0
    matches['control_feels_pain'] = 0
    matches['age_diff'] = 0
    for gender in pain_bg['sex'].unique():
        subs_temp = pain_bg.loc[pain_bg['sex'] == gender,'subid']
        subs = list(subs_temp)
        random.shuffle(subs)
        controls = controls_bg[controls_bg['sex']==gender].copy()
        for subject in subs:
            sub_age = pain_bg.loc[subject, 'age']
            ind = (np.abs(controls['age'] - sub_age)).argmin()
            age_diff = abs(controls.loc[ind, 'age'] - sub_age)
            matches.loc[subject, 'control_id':'age_diff'] = [ind, controls.loc[ind, 'sex'],
                                                             controls.loc[ind, 'age'],
                                                             controls.loc[ind, 'pain_now'],
                                                             controls.loc[ind, 'feels_pain'],
                                                             abs(controls.loc[ind, 'age'] - sub_age)]
            controls = controls.drop(ind, axis=0)

    sum_diff = sum(matches['age_diff'])
    n_too_big = matches[matches['age_diff']>age_diff_cutoff].shape[0]
    print('iteration ' + str(i) + ' age diff > ' + str(age_diff_cutoff) + ' years in ' + str(matches[matches['age_diff']>3].shape[0]) + ' subs, total sum of age diff ' + str(sum_diff))
    if i == 1:
        best_matches = matches.copy()
        min_sum = n_too_big
    elif (sum_diff < sum(best_matches.age_diff)) | (sum_diff < sum(best_matches.age_diff) & n_too_big < min_sum):
        best_matches = matches.copy()
        min_sum = n_too_big

print(best_matches[best_matches.age_diff>age_diff_cutoff])
# #
best_matches.to_csv('/m/nbe/scratch/socbrain/kipupotilaat/data/age_and_gender_matched_subs_pain_helsinki_after_qc.csv')
# #
# # ## Get second match per person
# rm_subid = best_matches['control_id']
# rm_indices = controls_bg[controls_bg['subid'].isin(rm_subid.tolist())].index
# #
# less_controls = controls_bg.drop(labels=rm_subid.tolist())
#
# age_diff_cutoff = 3
# min_sum = 1000
# for i in range(1,500):
#     matches = pain_bg.copy()
#     matches['control_id'] = 0
#     matches['control_sex'] = np.nan
#     matches['control_age'] = 0
#     matches['control_pain_now'] = 0
#     matches['control_feels_pain'] = 0
#     matches['age_diff'] = 0
#     for gender in pain_bg['sex'].unique():
#         subs_temp = pain_bg.loc[pain_bg['sex'] == gender,'subid']
#         subs = list(subs_temp)
#         random.shuffle(subs)
#         controls = less_controls[less_controls['sex']==gender].copy()
#         for subject in subs:
#             sub_age = pain_bg.loc[subject, 'age']
#             ind = (np.abs(controls['age'] - sub_age)).argmin()
#             age_diff = abs(controls.loc[ind, 'age'] - sub_age)
#             matches.loc[subject, 'control_id':'age_diff'] = [ind, controls.loc[ind, 'sex'],
#                                                              controls.loc[ind, 'age'],
#                                                              controls.loc[ind, 'pain_now'],
#                                                              controls.loc[ind, 'feels_pain'],
#                                                              abs(controls.loc[ind, 'age'] - sub_age)]
#             controls = controls.drop(ind, axis=0)
#
#     sum_diff = sum(matches['age_diff'])
#     n_too_big = matches[matches['age_diff']>age_diff_cutoff].shape[0]
#     print('iteration ' + str(i) + ' age diff > ' + str(age_diff_cutoff) + ' years in ' + str(matches[matches['age_diff']>3].shape[0]) + ' subs, total sum of age diff ' + str(sum_diff))
#     if i == 1:
#         best_matches = matches.copy()
#         min_sum = n_too_big
#     elif (sum_diff < sum(best_matches.age_diff)) | (sum_diff < sum(best_matches.age_diff) & n_too_big < min_sum):
#         best_matches = matches.copy()
#         min_sum = n_too_big
#
# print(best_matches[best_matches.age_diff>age_diff_cutoff])
# # #
# best_matches.to_csv('/m/nbe/scratch/socbrain/kipupotilaat/data/second_age_and_gender_matched_subs_pain_helsinki.csv')

# problem_subs = best_matches.loc[best_matches.age_diff>age_diff_cutoff, 'subid']
#
# #last_resort_controls = controls_small.loc[(controls_small.sex == '0') & (controls_small.feels_pain < 4) &
# #                                      (controls_small.pain_now < 4), :]
#
# #last_resort_controls.loc[(last_resort_controls.pain_now==3) | (last_resort_controls.feels_pain==3) | (last_resort_controls.pain_now==2) | (last_resort_controls.feels_pain==2)]
#
#
# # problem subs 10.5.
# # 2672 age 57
# # 4296 age 56
# # 5451 age 59
# # 9861 age 60
# # 9986 age 59