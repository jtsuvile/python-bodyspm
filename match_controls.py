from bodyfunctions import *
import numpy as np
import pandas as pd
import random



bg_pain_fibro = pd.read_csv('/Volumes/Shield1/kipupotilaat/data/stockholm/bg_pain_stockholm_fibro_2024_03_22.csv')
bg_pain_fibro['pain_condition'] = 'fibromyalgia'
bg_pain_clbp = pd.read_csv('/Volumes/Shield1/kipupotilaat/data/stockholm/bg_pain_stockholm_lbp_2024_03_22.csv')
bg_pain_clbp['pain_condition'] = 'clbp'

bg_pain = pd.concat([bg_pain_fibro, bg_pain_clbp])
pain_bg = bg_pain[['subid','age','sex','pain_now','pain_condition']]
csvname = '/Volumes/Shield1/kipupotilaat/data/stockholm/karolinska_matched_controls_2024_03_22.csv'


outcsvname = '/Volumes/Shield1/kipupotilaat/data/stockholm/bg_matched_controls_all_karolinska.csv'
bg_controls = pd.read_csv('/Users/juusu53/Documents/projects/kipupotilaat/data/controls/bg_all_controls_16_10_2020.csv')
keep_cols = ['subid','age','sex','pain_now','feels_pain','pain_chronic',
                                    'hist_abdomen', 'hist_back_shoulder', 'hist_headache', 'hist_joint_limb',
                                    'hist_menstrual', 'hist_migraine', 'bpi_now', 'bpi_average']

controls_bg = bg_controls[keep_cols]
#pain_bg = bg_pain[keep_cols]

controls_bg.set_index('subid', drop=False, inplace=True)
# TODO Update to current definitions

controls_bg = controls_bg[(controls_bg.pain_chronic == 0) &
                          ((controls_bg.bpi_now < 3) | (np.isnan(controls_bg.bpi_now))) &
                          ((controls_bg.bpi_average < 3) | (np.isnan(controls_bg.bpi_average))) &
                          ((controls_bg.feels_pain < 3) | (np.isnan(controls_bg.feels_pain))) &
                          ((controls_bg.hist_back_shoulder == 0) | (controls_bg.hist_back_shoulder == 3))].copy()

acceptable_controls_original = controls_bg.copy()

pain_bg.set_index('subid', inplace=True, drop=False)
#
age_diff_cutoff = 5
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
            ilocind = (np.abs(controls['age'] - sub_age)).argmin()
            ind = controls.iloc[ilocind]['subid']
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
