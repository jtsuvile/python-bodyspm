import os
import sys
import pandas as pd
from classdefinitions import Subject, Stimuli
from bodyfunctions import combine_data, preprocess_subjects, add_background_table
import matplotlib.pyplot as plt
import numpy as np
import time
import csv


who = 'helsinki_endo'
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
if who == 'helsinki_endo':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/subjects/'
    outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/processed/'
    subfile = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/subs.txt'
    field_names = [['sex', 'age', 'weight','height','handedness','education','work_physical','work_sitting','profession','psychologist','psychiatrist','neurologist'],
               ['pain_now','pain_last_day', 'pain_chronic','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns','hist_crps','hist_fibro'],
               ['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_fear','feels_surprise','feels_disgust'],
               ['bpi_worst', 'bpi_least', 'bpi_average', 'bpi_now', 'bpi_painkiller_relief'],
               ['bpi_functioning', 'bpi_mood','bpi_walk','bpi_work', 'bpi_relationships','bpi_sleep','bpi_enjoyment']]
    csvname = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/endometriosis_patients_26_05_2023.csv'
elif who == 'helsinki_endo_controls':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/subjects/'
    outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/matched_controls/'
    subfile = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/control_subs_painfree.txt'
    field_names = [
        ['sex', 'age', 'weight', 'height', 'handedness', 'education', 'work_physical', 'work_sitting', 'profession',
         'psychologist', 'psychiatrist', 'neurologist'],
        ['pain_now', 'pain_last_day', 'pain_chronic', 'hist_migraine', 'hist_headache', 'hist_abdomen',
         'hist_back_shoulder', 'hist_joint_limb', 'hist_menstrual',
         'painkillers_overcounter', 'painkillers_prescription', 'painkillers_othercns'],
        ['feels_pain', 'feels_depression', 'feels_anxiety', 'feels_happy', 'feels_sad', 'feels_angry', 'feels_fear',
         'feels_surprise', 'feels_disgust'],
        ['bpi_worst', 'bpi_least', 'bpi_average', 'bpi_now', 'bpi_painkiller_relief'],
        ['bpi_functioning', 'bpi_mood', 'bpi_walk', 'bpi_work', 'bpi_relationships', 'bpi_sleep', 'bpi_enjoyment']]
    csvname = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/endometriosis_controls_18_03_2022.csv'
else:
    print('wrong branch')

# if who is 'helsinki' or who is 'control':
with open(subfile) as f:
    subnums = f.readlines()
subnums = [x.strip() for x in subnums]

#subnums = [84002, 50436, 55419, 95848, 63165, 79662, 49764, 68328, 64110, 97405, 56216, 26270, 28358]
#
#
# # read subjects from web output and write out to a more sensible format
if who == 'helsinki' or who == 'stockholm' or who == 'helsinki_endo':
   preprocess_subjects(subnums, dataloc, outdataloc, stim, bg_files, field_names, intentionally_empty=True)
else:
   preprocess_subjects(subnums, dataloc, outdataloc, stim, bg_files, field_names)

# # # Gather subjects into one dict
# #
# # #grouping = [groupname] * len(subnums)
# #
#
# Combining data (with or without pain information)


print("combining data from ", len(subnums), " subjects")
print("getting started")
full_dataset = combine_data(outdataloc, subnums,
                           save=True, noImages=False)

bg = full_dataset['bg']
bg.to_csv(csvname)
#
# end = time.time()
# print(end - start)
#
