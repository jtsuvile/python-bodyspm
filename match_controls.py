
#from bodyfunctions import *
import pickle
import numpy as np
import pandas as pd
import random

dataloc_controls = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/'
datafile_controls = dataloc_controls + 'no_chronic_pain.pickle'

dataloc_pain = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/'
datafile_pain = dataloc_pain + 'full_dataset.pickle'

controls = pickle.load(open(datafile_controls, "rb"))
pain = pickle.load(open(datafile_pain, "rb"))
print('done loading in data')

controls_small = controls['bg'][['age','sex','pain_now','feels_pain']].copy()
controls_small['age'] = pd.to_numeric(controls_small['age'])
controls_small['pain_now'] = pd.to_numeric(controls_small['pain_now'])
controls_small['feels_pain'] = pd.to_numeric(controls_small['feels_pain'])

controls_small['subid'] = controls_small.index.values

pain_small = pain['bg'][['age','sex']].copy()
pain_small['age'] = pd.to_numeric(pain_small['age'])
pain_small['subid'] = pain_small.index.values

#
age_diff_cutoff = 3
min_sum = 1000
for i in range(1,500):
    matches = pain_small.copy()
    matches['control_id'] = 0
    matches['control_sex'] = np.nan
    matches['control_age'] = 0
    matches['control_pain_now'] = 0
    matches['control_feels_pain'] = 0
    matches['age_diff'] = 0
    for gender in pain_small['sex'].unique():
        subs = pain_small.loc[pain_small['sex'] == gender,'subid']
        random.shuffle(subs)
        # preferred control group is fully pain-free
        subs_control_pref = controls_small[(controls_small.sex == gender) & (controls_small.feels_pain == 0) &
                                           (controls_small.pain_now == 0)]
        # alternatively, subjects who are not fully pain free but have max 1 as their pain ratings
        subs_control_alt = controls_small[(controls_small.sex == gender) & (((controls_small.feels_pain == 1) &
                                           (controls_small.pain_now == 1)) | ((controls_small.feels_pain == 1) &
                                           (controls_small.pain_now == 0)) | ((controls_small.feels_pain == 0) &
                                           (controls_small.pain_now == 1)))]
        for subject in subs:
            sub_age = pain_small.loc[subject, 'age']
            ind = (np.abs(subs_control_pref['age'] - sub_age)).argmin()
            age_diff = abs(subs_control_pref.loc[ind, 'age'] - sub_age)
            better = 'norm'
            if age_diff > age_diff_cutoff:
                idx = (np.abs(subs_control_alt['age'] - sub_age)).argmin()
                alt_age_diff = abs(subs_control_alt.loc[idx, 'age'] - sub_age)
                if alt_age_diff < age_diff:
                    print('age diff between subject ' + subject + ' and ' + ind + ' was ' + str(
                        age_diff) + ' and alt diff was ' + str(alt_age_diff) + '. Using alt list.')
                    better = 'alt'
            if better == 'alt':
                matches.loc[subject, 'control_id':'age_diff'] = [idx,subs_control_alt.loc[idx, 'sex'],
                                                                 subs_control_alt.loc[idx, 'age'],
                                                                 subs_control_alt.loc[idx, 'pain_now'],
                                                                 subs_control_alt.loc[idx, 'feels_pain'],
                                                                 abs(subs_control_alt.loc[idx, 'age'] - sub_age)]
                subs_control_alt = subs_control_alt.drop(idx, axis=0)
            elif better == 'norm':
                matches.loc[subject, 'control_id':'age_diff'] = [ind, subs_control_pref.loc[ind, 'sex'],
                                                                 subs_control_pref.loc[ind, 'age'],
                                                                 subs_control_pref.loc[ind, 'pain_now'],
                                                                 subs_control_pref.loc[ind, 'feels_pain'],
                                                                 abs(subs_control_pref.loc[ind, 'age'] - sub_age)]
                subs_control_pref = subs_control_pref.drop(ind, axis=0)

    sum_diff = sum(matches['age_diff'])
    n_too_big = matches[matches['age_diff']>age_diff_cutoff].shape[0]
    print('iteration ' + str(i) + ' age diff > ' + str(age_diff_cutoff) + ' years in ' + str(matches[matches['age_diff']>3].shape[0]) + ' subs, total sum of age diff ' + str(sum_diff))
    if (n_too_big < min_sum) | (n_too_big == min_sum & sum_diff < sum(best_matches.age_diff)):
        best_matches = matches.copy()
        min_sum = n_too_big

print(best_matches[best_matches.age_diff>age_diff_cutoff])
#
best_matches.to_csv('/m/nbe/scratch/socbrain/kipupotilaat/data/age_and_gender_matched_subs_pain_helsinki.csv')
problem_subs = best_matches.loc[best_matches.age_diff>age_diff_cutoff, 'subid']

#last_resort_controls = controls_small.loc[(controls_small.sex == '0') & (controls_small.feels_pain < 4) &
#                                      (controls_small.pain_now < 4), :]

#last_resort_controls.loc[(last_resort_controls.pain_now==3) | (last_resort_controls.feels_pain==3) | (last_resort_controls.pain_now==2) | (last_resort_controls.feels_pain==2)]


# problem subs 10.5.
# 2672 age 57
# 4296 age 56
# 5451 age 59
# 9861 age 60
# 9986 age 59