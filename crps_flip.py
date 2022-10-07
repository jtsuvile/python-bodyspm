from bodyfunctions import *
import h5py
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

import sys

figloc = '/m/nbe/scratch/socbrain/kipupotilaat/figures/'
maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'
dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/crps/'
datafile = get_latest_datafile(dataloc)

mask_fb = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')
mask_one = read_in_mask(maskloc + 'mask_front_new.png')

stim_names = { 'pain_0': ['acute pain', 1], 'pain_1': ['chonic pain', 1], 'sensitivity_0': ['tactile sensitivity', 1],
               'sensitivity_1': ['nociceptive sensitivity', 1], 'sensitivity_2': ['hedonic sensitivity', 1]}


curr = mask_fb.copy()
right_front = curr[:, 1:85]
left_front = curr[:, 85:169]
left_back = curr[:, 172:256]
right_back = curr[:, 256:340]

# visualise sided parameters, make sure it looks as it should
# left_back.shape[1] - left_front.shape[1]
# right_back.shape[1] - right_front.shape[1]
#
# left = left_back + np.flip(left_front, axis=1)
# right = right_back + np.flip(right_front, axis=1)
#
# fig = plt.figure()
#
# ax1 = plt.subplot(121)
# img1 = plt.imshow(left)
#
# ax2 = plt.subplot(122)
# img2 = plt.imshow(right)
# plt.savefig(figloc + 'sides_combined_7.png')
# plt.close()

with h5py.File(datafile, 'r') as h:
    pain_chronic = h['pain_1'].value
    pain_acute = h['pain_0'].value
    subids = h['subid'].value

side_pain = np.zeros((1,subids.shape[0]))
crps_patients_with_side = np.vstack((subids, side_pain))

left_side = np.sum(np.sum(np.concatenate((pain_chronic[:,:, 85:169], pain_chronic[:,:, 172:256]),1),2),1)
right_side = np.sum(np.sum(np.concatenate((pain_chronic[:, 1:85], pain_chronic[:, 256:340]),1),2),1)

patient_pain_left = left_side - right_side > 0
patients_to_flip = subids[~patient_pain_left] # pick patients whose pain is typically on the right side & flip them over

# THIS IS NOT THE WAY TO DO THIS!! except for maybe emotion maps?
pain_chronic_flipped = np.vstack((np.flip(pain_chronic[~patient_pain_left, :, :], axis=2), pain_chronic[patient_pain_left,:,:]))
pain_acute_flipped = np.vstack((np.flip(pain_acute[~patient_pain_left, :, :], axis=2), pain_acute[patient_pain_left,:,:]))

# this is how to flip twosided maps
data_flipped = np.concatenate((np.flip(pain_chronic[patient_pain_left, :, 0:171], axis=2),
                               np.flip(pain_chronic[patient_pain_left, :, 170:341], axis=2)), axis=2)
data_crps = np.vstack((data_flipped, pain_chronic[~patient_pain_left, :, :]))