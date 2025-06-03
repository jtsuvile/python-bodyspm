import os
import sys
import pandas as pd
from classdefinitions import Subject, Stimuli
from bodyfunctions import make_qc_figures, preprocess_subjects, count_pixels_posneg
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


who = 'stockholm_test'

if who == 'control':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/subjects/'
    outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/qc/'
    subfile = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/subs.txt'
    csvname = '/m/nbe/scratch/socbrain/kipupotilaat/data/bg_all_controls.csv'
if who == 'stockholm_test':
    dataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm_old/test_sub_after_server_upgrade/subjects'
    outdataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm_old/test_sub_after_server_upgrade/qc/'
    subfile = '/Volumes/Shield1/kipupotilaat/data/stockholm_old/test_sub_after_server_upgrade/subs.txt'
    csvname = '/m/nbe/scratch/socbrain/kipupotilaat/data/bg_all_controls.csv'
elif who == 'stockholm_lbp':
    dataloc = '/Volumes/Shield1/backups_aalto_scratch/kipupotilaat/data/stockholm/subjects/'
    outdataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/processed/lbp_patients_qc/'
    subfile = '/Volumes/Shield1/kipupotilaat/data/stockholm/stockholm_subnums_lbp_pre_qc.txt'
elif who == 'stockholm_fibro':
    dataloc = '/Volumes/Shield1/backups_aalto_scratch/kipupotilaat/data/stockholm/subjects/'
    outdataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/processed/fibro_patients_qc/'
    subfile = '/Volumes/Shield1/kipupotilaat/data/stockholm/stockholm_subnums_fibro_pre_qc.txt'

subnums = list(pd.read_csv(subfile, header=None)[0])
onesided = [True, True, True, True, True, True, True, False, False, False, False, False]
data_names = ['emotions_0', 'emotions_1', 'emotions_2', 'emotions_3', 'emotions_4','emotions_5','emotions_6',
              'sensitivity_0', 'sensitivity_1', 'sensitivity_2', 'pain_0', 'pain_1']
display_names = ['sadness', 'happiness', 'anger', 'surprise', 'fear', 'disgust', 'neutral',
 'tactile sensitivity','nociceptive sensitivity', 'hedonic sensitivity','current pain', 'chonic pain']
stim = Stimuli(data_names, onesided=onesided, show_names = display_names)

make_qc_figures(subnums, dataloc, stim, outdataloc)
# # NB: square for marking intentionally empty bodies approximately at
# # [530:580,430:480] in the full image
