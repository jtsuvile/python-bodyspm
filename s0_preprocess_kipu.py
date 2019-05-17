import os
import sys
import pandas as pd
from classdefinitions import Subject, Stimuli
from bodyfunctions import combine_data, preprocess_subjects
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import codecs

who = 'control'

start = time.time()
# set up stimuli description
onesided = [True, True, True, True, True, True, True, False, False, False, False, False]
# boolean or list of booleans describing if data is onesided (e.g. emotion body maps, with one image
# representing intensifying and one image representing lessening activation. In this case, one side is deducted from
# the other. Alternative (False) describes situation where both sides of colouring are retained, e.g. touch allowances
# for front and back of body.
data_names = ['emotions_0', 'emotions_1', 'emotions_2', 'emotions_3', 'emotions_4','emotions_5','emotions_6', 'sensitivity_0','sensitivity_1','sensitivity_2', 'pain_0', 'pain_1']
stim_names = ['stim1','stim2','stim3','stim4','stim5', 'pain1', 'pain2'] # potentially add stimulus names for more intuitive data handling
bg_files = ['data.txt','pain_info.txt','current_feelings.txt','BPI_1.txt','BPI_2.txt']
field_names = [['sex', 'age', 'weight','height','handedness','education','work_physical','work_sitting','psychologist','psychiatrist', 'neurologist'],
               ['pain_now','pain_last_day', 'pain_chronic','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual','hist_crps','hist_fibro',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns'],
               ['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_surprise','feels_disgust'],
               ['bpi_worst', 'bpi_least', 'bpi_average', 'bpi_now', 'bpi_painkiller_relief'],
               ['bpi_functioning', 'bpi_mood','bpi_walk','bpi_work', 'bpi_relationships','bpi_sleep','bpi_enjoyment']]
field_names_control = [['sex', 'age', 'weight','height','handedness','education','work_physical','work_sitting','psychologist','psychiatrist', 'neurologist'],
               ['pain_now','pain_last_day', 'pain_chronic','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns'],
               ['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_surprise','feels_disgust'],
               ['bpi_worst', 'bpi_least', 'bpi_average', 'bpi_now', 'bpi_painkiller_relief'],
               ['bpi_functioning', 'bpi_mood','bpi_walk','bpi_work', 'bpi_relationships','bpi_sleep','bpi_enjoyment']]


# define stimulus set
stim = Stimuli(data_names, onesided=onesided)

# inputs
if who == 'control':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/subjects/'
    outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/'
    subfile = '/m/nbe/scratch/socbrain/kipupotilaat/data/matched_controls_may_2019.txt'
elif who == 'helsinki':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/subjects/'
    outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/'
    subfile = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/kipu_subs.txt'

subnums = []
with open(subfile) as f:
    subnums = f.readlines()

subnums = [x.strip() for x in subnums]

print("hiya")

#subnums_left = [3200, 6058, 6460]
# # read subjects from web output and write out to a more sensible format
#preprocess_subjects(subnums_left, dataloc, outdataloc, stim, bg_files, field_names_control)
#
# # Gather subjects into one dict
#
# #grouping = [groupname] * len(subnums)


full_dataset = combine_data(outdataloc, subnums, save=True)

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

# NB: 3669 not correct subject number