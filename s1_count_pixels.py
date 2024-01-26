import os
import sys
import pandas as pd
from bodyfunctions import *
import numpy as np
import csv


bgdatapath = '/Volumes/Shield1/kipupotilaat/data/stockholm/bg_pain_stockholm_lbp_26_01_2024.csv'
dataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/processed/lbp/'
outfilename ='/Volumes/Shield1/kipupotilaat/data/stockholm/bg_pain_stockholm_lbp_with_activations_01_2024.csv'

# bgdatapath = '/Volumes/Shield1/kipupotilaat/data/stockholm/bg_pain_stockholm_fibro_26_01_2024.csv'
# dataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/processed/fibro/'
# outfilename ='/Volumes/Shield1/kipupotilaat/data/stockholm/bg_pain_stockholm_fibro_with_activations_01_2024.csv'


# bgdatapath = '/m/nbe/scratch/socbrain/kipupotilaat/data/bg_all_controls_16_10_2020.csv'
# dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/'
# outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/data/all_healthy_with_activations_29_10_2020.csv'

maskloc = '/Users/juusu53/Documents/projects/kipupotilaat/python_code/sample_data/'
datafile = get_latest_datafile(dataloc)

mask_one = read_in_mask(maskloc + 'mask_front_new.png')
mask_fb = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')

stim_names = {'emotions_0': ['sadness', 0], 'emotions_1': ['happiness', 0], 'emotions_2': ['anger', 0],
              'emotions_3': ['surprise', 0], 'emotions_4': ['fear', 0], 'emotions_5': ['disgust', 0],
              'emotions_6': ['neutral', 0],
              'pain_0': ['current_pain', 1], 'pain_1': ['chonic_pain', 1], 'sensitivity_0': ['tactile_sensitivity', 1],
              'sensitivity_1': ['nociceptive_sensitivity', 1], 'sensitivity_2': ['hedonic_sensitivity', 1]}

bg = pd.read_csv(bgdatapath)

for j, cond in enumerate(stim_names.keys()):
    with h5py.File(datafile, 'r') as h:
        data = h[cond][0] # TODO: something off here, must check
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


bg.to_csv(outfilename, na_rep='NaN')