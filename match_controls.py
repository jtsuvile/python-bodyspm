from bodyfunctions import *
import numpy as np
import pandas as pd
import random



bg_pain = pd.read_csv('/Volumes/Shield1/kipupotilaat/data/bg_pain_stockholm_fibro_19_01_2024.csv')

outcsvname = '/Volumes/Shield1/kipupotilaat/data/stockholm/age_and_gender_matched_subs_fibro_KI_19-01-2024.csv'
bg_controls = pd.read_csv('/Users/juusu53/Documents/projects/kipupotilaat/data/controls/bg_all_controls_16_10_2020.csv')
keep_cols = ['subid','age','sex','pain_now','feels_pain','pain_chronic',
                                    'hist_abdomen', 'hist_back_shoulder', 'hist_headache', 'hist_joint_limb',
                                    'hist_menstrual', 'hist_migraine', 'bpi_now', 'bpi_average']

controls_bg = bg_controls[keep_cols]
pain_bg = bg_pain[keep_cols]

controls_bg.set_index('subid', drop=False, inplace=True)
controls_bg = controls_bg[(controls_bg.pain_chronic == 0) & (controls_bg.pain_now == 0) &
                          (controls_bg.feels_pain < 3)].copy()

acceptable_controls_original = controls_bg.copy()

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
    matches['bpi_now'] = np.nan
    matches['bpi_average'] = np.nan
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
                                                             age_diff]
            #print(controls.loc[ind, 'bpi_now'])
            if pd.notna(controls.loc[ind, 'bpi_now']):
                 matches.loc[subject, 'bpi_now'] = controls.loc[ind, 'bpi_now']
                 matches.loc[subject, 'bpi_average'] = controls.loc[ind, 'bpi_average']
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
best_matches.to_csv(csvname)
