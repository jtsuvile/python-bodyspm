import os
import sys
import pandas as pd
from classdefinitions import Subject, Stimuli
from bodyfunctions import combine_data, preprocess_subjects, add_background_table
import matplotlib.pyplot as plt
import numpy as np
import time
import csv


who = 'controls'
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
if who == 'controls':
    dataloc = '/Volumes/Shield1/kipupotilaat/data/controls/subjects/'
    outdataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/controls/test/'
    subfile = '/Users/juusu53/Documents/projects/kipupotilaat/data/KI/controls_all_acceptable_nopain_karolinska.txt'
    field_names = [['sex', 'age', 'weight','height','handedness','education','work_physical','work_sitting','profession','psychologist','psychiatrist', 'neurologist'],
               ['pain_now','pain_last_day', 'pain_chronic','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns'],
               ['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_fear','feels_surprise','feels_disgust'],
               ['bpi_worst', 'bpi_least', 'bpi_average', 'bpi_now', 'bpi_painkiller_relief'],
               ['bpi_functioning', 'bpi_mood','bpi_walk','bpi_work', 'bpi_relationships','bpi_sleep','bpi_enjoyment']]
    csvname = '/Volumes/Shield1/kipupotilaat/data/stockholm/controls/test/bg_all_acceptable_controls_2025-02-19.csv'
elif who == 'stockholm':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/stockholm/subjects/'
    outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/stockholm/processed/'
    subfile = '/m/nbe/scratch/socbrain/kipupotilaat/data/stockholm/stockholm_patients.csv'
    subdata = pd.read_csv(subfile, sep=';')
    subnums = list(subdata['subid'])
    field_names = [['sex', 'age', 'weight','height','handedness','education','work_physical','work_sitting','psychologist','psychiatrist','neurologist'],
               ['pain_now','pain_last_day', 'pain_chronic','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns'],
               ['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_fear','feels_surprise','feels_disgust'],
               ['bpi_worst', 'bpi_least', 'bpi_average', 'bpi_now', 'bpi_painkiller_relief'],
               ['bpi_functioning', 'bpi_mood','bpi_walk','bpi_work', 'bpi_relationships','bpi_sleep','bpi_enjoyment']]
    csvname = '/m/nbe/scratch/socbrain/kipupotilaat/data/bg_pain_stockholm_2024-01-05.csv'
elif who == 'stockholm_lbp':
    dataloc = '/Volumes/Shield1/backups_aalto_scratch/kipupotilaat/data/stockholm/subjects/'
    outdataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/processed/lbp/'
    subfile = '/Volumes/Shield1/kipupotilaat/data/stockholm/stockholm_subnums_lbp_after_all_qc_steps.txt'
    field_names = [['sex', 'age', 'weight','height','handedness','education','work_physical','work_sitting','psychologist','psychiatrist','neurologist'],
               ['pain_now','pain_last_day', 'pain_chronic','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns'],
               ['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_fear','feels_surprise','feels_disgust'],
               ['bpi_worst', 'bpi_least', 'bpi_average', 'bpi_now', 'bpi_painkiller_relief'],
               ['bpi_functioning', 'bpi_mood','bpi_walk','bpi_work', 'bpi_relationships','bpi_sleep','bpi_enjoyment']]
    csvname = '/Volumes/Shield1/kipupotilaat/data/stockholm/bg_pain_stockholm_lbp_2024_03_22.csv'
elif who == 'stockholm_fibro':
    dataloc = '/Volumes/Shield1/backups_aalto_scratch/kipupotilaat/data/stockholm/subjects/'
    outdataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/processed/fibro/'
    subfile = '/Volumes/Shield1/kipupotilaat/data/stockholm/stockholm_subnums_fibro_after_all_qc_steps.txt'
    field_names = [['sex', 'age', 'weight','height','handedness','education','work_physical','work_sitting','psychologist','psychiatrist','neurologist'],
               ['pain_now','pain_last_day', 'pain_chronic','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns'],
               ['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_fear','feels_surprise','feels_disgust'],
               ['bpi_worst', 'bpi_least', 'bpi_average', 'bpi_now', 'bpi_painkiller_relief'],
               ['bpi_functioning', 'bpi_mood','bpi_walk','bpi_work', 'bpi_relationships','bpi_sleep','bpi_enjoyment']]
    csvname = '/Volumes/Shield1/kipupotilaat/data/stockholm/bg_pain_stockholm_fibro_2024_03_22.csv'
elif who == 'all_controls_stockholm':
    dataloc = '/Volumes/Shield1/backups_aalto_scratch/kipupotilaat/data/controls/subjects/'
    outdataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/controls/all/'
    subfile = '/Volumes/Shield1/kipupotilaat/data/stockholm/all_stockholm_controls_2024-03-22.txt'
    field_names = [['sex', 'age', 'weight','height','handedness','education','work_physical','work_sitting','profession','psychologist','psychiatrist', 'neurologist'],
               ['pain_now','pain_last_day', 'pain_chronic','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns'],
               ['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_fear','feels_surprise','feels_disgust'],
               ['bpi_worst', 'bpi_least', 'bpi_average', 'bpi_now', 'bpi_painkiller_relief'],
               ['bpi_functioning', 'bpi_mood','bpi_walk','bpi_work', 'bpi_relationships','bpi_sleep','bpi_enjoyment']]
    csvname = '/Volumes/Shield1/kipupotilaat/data/stockholm/bg_all_matched_controls_stockholm_2024_03_22.csv'
elif who == 'clbp_controls_stockholm':
    dataloc = '/Volumes/Shield1/backups_aalto_scratch/kipupotilaat/data/controls/subjects/'
    outdataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/controls/clbp/'
    subfile = '/Volumes/Shield1/kipupotilaat/data/stockholm/clbp_controls_2024-03-22.txt'
    field_names = [['sex', 'age', 'weight','height','handedness','education','work_physical','work_sitting','profession','psychologist','psychiatrist', 'neurologist'],
               ['pain_now','pain_last_day', 'pain_chronic','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns'],
               ['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_fear','feels_surprise','feels_disgust'],
               ['bpi_worst', 'bpi_least', 'bpi_average', 'bpi_now', 'bpi_painkiller_relief'],
               ['bpi_functioning', 'bpi_mood','bpi_walk','bpi_work', 'bpi_relationships','bpi_sleep','bpi_enjoyment']]
    csvname = '/Volumes/Shield1/kipupotilaat/data/stockholm/bg_clbp_matched_controls_stockholm_2024_03_22.csv'
elif who == 'fibro_controls_stockholm':
    dataloc = '/Volumes/Shield1/backups_aalto_scratch/kipupotilaat/data/controls/subjects/'
    outdataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/controls/fibro/'
    subfile = '/Volumes/Shield1/kipupotilaat/data/stockholm/fibromyalgia_controls_2024-03-22.txt'
    field_names = [['sex', 'age', 'weight','height','handedness','education','work_physical','work_sitting','profession','psychologist','psychiatrist', 'neurologist'],
               ['pain_now','pain_last_day', 'pain_chronic','hist_migraine','hist_headache','hist_abdomen','hist_back_shoulder','hist_joint_limb','hist_menstrual',
                'painkillers_overcounter','painkillers_prescription', 'painkillers_othercns'],
               ['feels_pain','feels_depression','feels_anxiety','feels_happy','feels_sad','feels_angry','feels_fear','feels_surprise','feels_disgust'],
               ['bpi_worst', 'bpi_least', 'bpi_average', 'bpi_now', 'bpi_painkiller_relief'],
               ['bpi_functioning', 'bpi_mood','bpi_walk','bpi_work', 'bpi_relationships','bpi_sleep','bpi_enjoyment']]
    csvname = '/Volumes/Shield1/kipupotilaat/data/stockholm/bg_fibromyalgia_matched_controls_stockholm_2024_03_22.csv'


with open(subfile) as f:
    subnums = f.readlines()
subnums = [x.strip() for x in subnums]
#
#
# # read subjects from web output and write out to a more sensible format
if who in ['helsinki','stockholm','stockholm_lbp','stockholm_fibro']:
    preprocess_subjects(subnums, dataloc, outdataloc, stim, bg_files, field_names, intentionally_empty=True)
else:
    preprocess_subjects(subnums, dataloc, outdataloc, stim, bg_files, field_names)

# Combining data 

print("combining data from ", len(subnums), " subjects")
print("getting started")
full_dataset = combine_data(outdataloc, subnums,
                            save=True, noImages=False)
#
bg = full_dataset['bg']
bg.to_csv(csvname)
#
# end = time.time()
# print(end - start)
#
