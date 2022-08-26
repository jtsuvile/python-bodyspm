import os
import sys
import pandas as pd
from bodyfunctions import *
import numpy as np
import csv

# bgdatapath = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/endometriosis_patients_18_03_2022.csv'
# dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/processed/'
# outfilename ='/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/endometriosis_with_activations_03_2022.csv'

bgdatapath = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/endometriosis_controls_18_03_2022.csv'
dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/matched_controls/'
outfilename ='/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/endometriosis_controls_with_activations_03_2022.csv'

maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'
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


bg.to_csv(outfilename, na_rep='NaN')


