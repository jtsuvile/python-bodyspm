from bodyfunctions import *
import h5py
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image

figloc = '/m/nbe/scratch/socbrain/kipupotilaat/figures/'
maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'
dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/'
datafile = get_latest_datafile(dataloc)


stim_names = {'emotions_0': ['sadness', 0], 'emotions_1': ['happiness', 0], 'emotions_2': ['anger', 0],
              'emotions_3': ['surprise', 0], 'emotions_4': ['fear', 0], 'emotions_5': ['disgust', 0],
              'emotions_6': ['neutral', 0],
              'pain_0': ['acute pain', 1], 'pain_1': ['chonic_pain', 1], 'sensitivity_0': ['tactile sensitivity', 1],
              'sensitivity_1': ['nociceptive sensitivity', 1], 'sensitivity_2': ['hedonic sensitivity', 1]}
color_defs = {'head': 26, 'shoulder': 128, 'arm': 102, 'hand': 204, 'torso': 51, 'crotch': 128, 'leg': 153, 'foot': 230}

rois_base = scipy.misc.imread(maskloc + 'ROIs_pain.png', flatten=True, mode='L')
outline_front = scipy.misc.imread(maskloc + 'outline_front.png', flatten=True, mode='L')
outline_back = scipy.misc.imread(maskloc + 'outline_back.png', flatten=True, mode='L')

cond = 'sensitivity_2'


if (stim_names[cond][1] == 0):
    mask = read_in_mask(maskloc + 'mask_front_new.png')
    rois = rois_base.copy()
else:
    mask = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')
    rois = np.hstack((rois_base, rois_base))


with h5py.File(datafile, 'r') as h:
    kipu = h[cond].value

# arm_color = 102

outline_back_better = outline_back.copy()
outline_back_better[outline_back_better <= 20] = 0
outline_back_better[outline_back_better > 20] = 1
outline_front_better = outline_front.copy()
outline_front_better[outline_front_better <= 20] = 0
outline_front_better[outline_front_better > 20] = 1



## sanity checks for masks etc
# scipy.misc.imsave(maskloc + 'outline_binary.png', outline_back_better)

# sensitivity_all = np.sum(kipu, axis=0)
#
# rois_with_outline = sensitivity_all + np.hstack((outline_front_better, outline_back_better))
#
# plt.imshow(rois_with_outline)
# plt.colorbar()
# plt.savefig(maskloc + 'data_with_outline_twosided.png')
# scipy.misc.imsave(maskloc + 'roi_with_outline_twosided.png', rois)
#
plt.hist(kipu.flatten())
plt.savefig(maskloc + 'hedonic_sensitivity_values.png')
# # plt.hist(outline_back)
# # plt.savefig(maskloc + 'outline_back_values.png')
#
# rois[rois!=color_defs['head']] = 0
# scipy.misc.imsave(maskloc + 'roi_test_head.png', rois)

