import os
import sys
import pandas as pd
from classdefinitions import Subject, Stimuli
from bodyfunctions import make_qc_figures
import matplotlib.pyplot as plt
import numpy as np
import time
import csv

who = 'matched_controls_helsinki'

if who == 'control':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/subjects/'
    outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/qc/'
    subfile = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/subs.txt'
    csvname = '/m/nbe/scratch/socbrain/kipupotilaat/data/bg_all_controls.csv'
elif who == 'helsinki':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/subjects/'
    outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/qc/'
    subfile = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/kipu_subs.txt'
    csvname = '/m/nbe/scratch/socbrain/kipupotilaat/data/all_pain_patients_21_10_2019.csv'
elif who == 'stockholm':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/stockholm/subjects/'
    outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/stockholm/qc/'
    subfile = '/m/nbe/scratch/socbrain/kipupotilaat/data/stockholm/subs.txt'
    csvname = '/m/nbe/scratch/socbrain/kipupotilaat/data/bg_pain_stockholm.csv'
elif who == 'matched_controls_helsinki':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/subjects/'
    outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/qc/'
    subfile = '/m/nbe/scratch/socbrain/kipupotilaat/data/age_and_gender_matched_subs_pain_helsinki.csv'
    csvname = '/m/nbe/scratch/socbrain/kipupotilaat/data/bg_matched_controls_30_10_2019.csv'
    matchdata = pd.read_csv(subfile)
    subnums = [str(x) for x in list(matchdata['control_id'])]


onesided = [True, True, True, True, True, True, True, False, False, False, False, False]
data_names = ['emotions_0', 'emotions_1', 'emotions_2', 'emotions_3', 'emotions_4','emotions_5','emotions_6',
              'sensitivity_0', 'sensitivity_1', 'sensitivity_2', 'pain_0', 'pain_1']

stim = Stimuli(data_names, onesided=onesided)

if who != 'matched_controls_helsinki' and who != 'matched_controls_two_each':
    print('re-reading')
    with open(subfile) as f:
        subnums = f.readlines()
    subnums = [x.strip() for x in subnums]

subnums=subnums[80:135]

make_qc_figures(subnums, dataloc, outdataloc, stim)