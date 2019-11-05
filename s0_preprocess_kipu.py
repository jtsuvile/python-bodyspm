import os
import sys
import pandas as pd
from classdefinitions import Subject, Stimuli
from bodyfunctions import combine_data, preprocess_subjects
import matplotlib.pyplot as plt
import numpy as np
import time
import csv


who = 'matched_controls_two_each'

start = time.time()
# set up stimuli description
onesided = [True, True, True, True, True, True, True, False, False, False, False, False]
# boolean or list of booleans describing if data is onesided (e.g. emotion body maps, with one image
# representing intensifying and one image representing lessening activation. In this case, one side is deducted from
# the other. Alternative (False) describes situation where both sides of colouring are retained, e.g. touch allowances
# for front and back of body.
data_names = ['emotions_0', 'emotions_1', 'emotions_2', 'emotions_3', 'emotions_4','emotions_5','emotions_6',
              'sensitivity_0', 'sensitivity_1', 'sensitivity_2', 'pain_0', 'pain_1']
bg_files = ['data.txt', 'pain_info.txt', 'current_feelings.txt', 'BPI_1.txt', 'BPI_2.txt']

# define stimulus set
stim = Stimuli(data_names, onesided=onesided)

# inputs
if who == 'control':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/subjects/'
    outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/'
    subfile = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/subs.txt'
    #subfile = '/m/nbe/scratch/socbrain/kipupotilaat/data/matched_controls_may_2019.txt'
    field_names = [['sex', 'age', 'weight','height','handedness','education','work_physical','work_sitting','profession','psychologist','psychiatrist', 'neurologist'],
               ['pain_now','pain_last_day', 'pain_chronic','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns'],
               ['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_fear','feels_surprise','feels_disgust'],
               ['bpi_worst', 'bpi_least', 'bpi_average', 'bpi_now', 'bpi_painkiller_relief'],
               ['bpi_functioning', 'bpi_mood','bpi_walk','bpi_work', 'bpi_relationships','bpi_sleep','bpi_enjoyment']]
elif who == 'helsinki':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/subjects/'
    outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/'
    subfile = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/kipu_subs.txt'
    grouping_file = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/kipuklinikka_kiputyyppi.csv'
    field_names = [['sex', 'age', 'weight','height','handedness','education','work_physical','work_sitting','profession','psychologist','psychiatrist','neurologist'],
               ['pain_now','pain_last_day', 'pain_chronic','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns','hist_crps','hist_fibro'],
               ['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_fear','feels_surprise','feels_disgust'],
               ['bpi_worst', 'bpi_least', 'bpi_average', 'bpi_now', 'bpi_painkiller_relief'],
               ['bpi_functioning', 'bpi_mood','bpi_walk','bpi_work', 'bpi_relationships','bpi_sleep','bpi_enjoyment']]
elif who == 'matched_controls':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/subjects/'
    outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/'
    subfile = '/m/nbe/scratch/socbrain/kipupotilaat/data/age_and_gender_matched_subs_pain_helsinki.csv'
    field_names = [['sex', 'age', 'weight','height','handedness','education','work_physical','work_sitting','profession','psychologist','psychiatrist', 'neurologist'],
               ['pain_now','pain_last_day', 'pain_chronic','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns'],
               ['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_fear','feels_surprise','feels_disgust'],
               ['bpi_worst', 'bpi_least', 'bpi_average', 'bpi_now', 'bpi_painkiller_relief'],
               ['bpi_functioning', 'bpi_mood','bpi_walk','bpi_work', 'bpi_relationships','bpi_sleep','bpi_enjoyment']]
    matchdata = pd.read_csv(subfile)
    subnums = list(matchdata['control_id'])
elif who == 'matched_controls_two_each':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/subjects/'
    outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/'
    subfile1 = '/m/nbe/scratch/socbrain/kipupotilaat/data/age_and_gender_matched_subs_pain_helsinki.csv'
    subfile2 = '/m/nbe/scratch/socbrain/kipupotilaat/data/second_age_and_gender_matched_subs_pain_helsinki.csv'
    field_names = [['sex', 'age', 'weight','height','handedness','education','work_physical','work_sitting','profession','psychologist','psychiatrist', 'neurologist'],
               ['pain_now','pain_last_day', 'pain_chronic','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns'],
               ['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_fear','feels_surprise','feels_disgust'],
               ['bpi_worst', 'bpi_least', 'bpi_average', 'bpi_now', 'bpi_painkiller_relief'],
               ['bpi_functioning', 'bpi_mood','bpi_walk','bpi_work', 'bpi_relationships','bpi_sleep','bpi_enjoyment']]
    matchdata1 = pd.read_csv(subfile1)
    matchdata2 = pd.read_csv(subfile2)
    matchdata = pd.concat([matchdata1, matchdata2])
    subnums = list(matchdata['control_id'])

# if who is not 'matched_controls':
#     with open(subfile) as f:
#         subnums = f.readlines()
#     subnums = [x.strip() for x in subnums]
#
# print("hiya")

# subnums = ['4308']
# read subjects from web output and write out to a more sensible format
# preprocess_subjects(subnums, dataloc, outdataloc, stim, bg_files, field_names)

# # Gather subjects into one dict
#
# #grouping = [groupname] * len(subnums)
#
if who == 'helsinki':
    with open(grouping_file, newline='', encoding='utf-8-sig') as csvfile:
        grouping_data = list(csv.reader(csvfile, delimiter=';'))
    group_df = pd.DataFrame(grouping_data)
    group_df.columns =['diagnosis_1', 'diagnosis_2', 'subid']
    group_df['diagnosis_1'] = group_df['diagnosis_1'].str.upper()
    full_dataset = combine_data(outdataloc, group_df['subid'].tolist(), groups=group_df['diagnosis_1'].tolist(),
                                save=True, noImages=False) #, noImages=True
    bg = full_dataset['bg']
    bg.to_csv('/m/nbe/scratch/socbrain/kipupotilaat/data/all_pain_patients_21_10_2019.csv')
elif who == 'control':
    print("combining data from ", len(subnums), " subjects")
    full_dataset = combine_data(outdataloc, subnums,
                                save=False, noImages=True)
    bg = full_dataset['bg']
    bg.to_csv('/m/nbe/scratch/socbrain/kipupotilaat/data/bg_all_controls.csv')
elif who == 'matched_controls':
    print("combining data from ", len(subnums), " subjects")
    full_dataset = combine_data(outdataloc, subnums,
                                save=True, noImages=False)
    bg = full_dataset['bg']
    bg.to_csv('/m/nbe/scratch/socbrain/kipupotilaat/data/bg_matched_controls_30_10_2019.csv')
elif who == 'matched_controls_two_each':
    print("combining data from ", len(subnums), " subjects")
    full_dataset = combine_data(outdataloc, subnums,
                                save=True, noImages=False)
    bg = full_dataset['bg']
    bg.to_csv('/m/nbe/scratch/socbrain/kipupotilaat/data/bg_double_matched_controls_30_10_2019.csv')

end = time.time()
print(end - start)

## code for reading in grouping (e.g. pain type) from file
# with open('/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/kipuklinikka_kiputyyppi.csv', mode='r', encoding="utf-8-sig") as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=';')
#     mydict = {row[1]:row[0] for row in csv_reader}
#
# grouping = []
# for subject in subnums:
#     if subject in mydict:
#         grouping.append(mydict[subject])
#     else:
#         subnums.remove(subject)

# # select patients with no history of chronic pain
# noChronicSubs = [key  for (key, value) in full_dataset['bg']['pain_chronic'].items() if value == '0']
# noPainNowSubs = [key  for (key, value) in full_dataset['bg']['pain_now'].items() if value == '0']
# noPainFeelsSubs = [key  for (key, value) in full_dataset['bg']['feels_pain'].items() if value == '0']
# noAcuteSubs = list(set(noPainNowSubs) & set(noPainFeelsSubs))
# noPainSubs = list(set(noChronicSubs) & set(noPainNowSubs) & set(noPainFeelsSubs))
#
# with open('/m/nbe/scratch/socbrain/kipupotilaat/data/controls/noPainSubs.txt', 'w') as f:
#     for item in noPainSubs:
#         f.write("%s\n" % item)
#
#
# with open('/m/nbe/scratch/socbrain/kipupotilaat/data/controls/noAcutePainSubs.txt', 'w') as f:
#     for item in noAcuteSubs:
#         f.write("%s\n" % item)
#
#
# with open('/m/nbe/scratch/socbrain/kipupotilaat/data/controls/noChronicPainSubs.txt', 'w') as f:
#     for item in noChronicSubs:
#         f.write("%s\n" % item)
#

