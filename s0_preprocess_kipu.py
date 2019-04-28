import os
import sys
import pandas as pd
from classdefinitions import Subject, Stimuli
from bodyfunctions import combine_data, preprocess_subjects
import matplotlib.pyplot as plt
import numpy as np
import time

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

# inputs
dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/subjects/'
outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/'
subnums = []
with open('/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/kipu_subs.txt') as f:
    subnums = f.readlines()

subnums = [x.strip() for x in subnums]

# define stimulus set
stim = Stimuli(data_names, onesided=onesided)

# read subjects from web output and write out to a more sensible format
preprocess_subjects(subnums, dataloc, outdataloc, stim, bg_files, field_names)

# Gather subjects into one dict

full_dataset = combine_data(outdataloc, subnums, save=True)

end = time.time()
print(end - start)
