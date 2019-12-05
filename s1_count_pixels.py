import os
import sys
import pandas as pd
from classdefinitions import Subject, Stimuli
from bodyfunctions import *
import matplotlib.pyplot as plt
import numpy as np
import time
import csv



# bgdatapath = '/m/nbe/scratch/socbrain/kipupotilaat/data/all_pain_patients_12_08_2019.csv'
# dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/'
# outfilename ='/m/nbe/scratch/socbrain/kipupotilaat/data/all_pain_patients_with_activations.csv'

# bgdatapath = '/m/nbe/scratch/socbrain/kipupotilaat/data/bg_matched_controls.csv'
# dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/matched_controls/'
# outfilename ='/m/nbe/scratch/socbrain/kipupotilaat/data/matched_controls_with_activations.csv'

bgdatapath = '/m/nbe/scratch/socbrain/kipupotilaat/data/bg_all_controls.csv'
dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/'
outfilename ='/m/nbe/scratch/socbrain/kipupotilaat/data/all_healthy_with_activations.csv'

maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'
datafile = get_latest_datafile(dataloc)

mask_one = read_in_mask(maskloc + 'mask_front_new.png')
mask_fb = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')

stim_names = {'emotions_0': ['sadness', 0], 'emotions_1': ['happiness', 0], 'emotions_2': ['anger', 0],
              'emotions_3': ['surprise', 0], 'emotions_4': ['fear', 0], 'emotions_5': ['disgust', 0],
              'emotions_6': ['neutral', 0],
              'pain_0': ['acute pain', 1], 'pain_1': ['chonic_pain', 1], 'sensitivity_0': ['tactile sensitivity', 1],
              'sensitivity_1': ['nociceptive sensitivity', 1], 'sensitivity_2': ['hedonic sensitivity', 1]}

stim_names_emotions = {'emotions_0':'sadness', 'emotions_1':'happiness', 'emotions_2':'anger', 'emotions_3':'surprise',
              'emotions_4': 'fear', 'emotions_5':'disgust', 'emotions_6':'neutral'}

stim_names_pain = {'pain_0':'acute pain', 'pain_1': 'chonic_pain'}
stim_names_sensitivity = {'sensitivity_0':'tactile sensitivity',
              'sensitivity_1':'nociceptive sensitivity', 'sensitivity_2':'hedonic sensitivity'}

bg = pd.read_csv(bgdatapath)

for j, cond in enumerate(stim_names.keys()):
    with h5py.File(datafile, 'r') as h:
        data = h[cond].value
        ## yes I have double checked that the subjects are in the same order in csv and in hdf5 file
        #subids = h['subid'].value
    if stim_names[cond][1] == 1:
        mask_use = mask_fb
    else:
        mask_use = mask_one
    pos_n, pos_prop, neg_n, neg_prop = count_pixels_posneg(data, mask_use)
    bg[cond + '_pos_color'] = pos_prop
    if stim_names[cond][1] == 0:
        bg[cond + '_neg_color'] = neg_prop


bg.to_csv(outfilename)