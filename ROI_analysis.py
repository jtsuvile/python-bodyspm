from bodyfunctions import *
import h5py
import numpy as np
import imageio
#import matplotlib.pyplot as plt
#from PIL import Image
from operator import add

# dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/matched_controls/'
# bgdatapath = '/m/nbe/scratch/socbrain/kipupotilaat/data/bg_matched_controls_18_11_2020.csv'
# outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/data/matched_controls_with_activations_by_roi_01_2021.csv'

bgdatapath = '/m/nbe/scratch/socbrain/kipupotilaat/data/all_pain_patients_15_10_2020.csv'
dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/'
outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/data/all_pain_patients_with_activations_by_roi_01_2021.csv'

datafile = get_latest_datafile(dataloc)
maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'

stim_names = {'emotions_0': ['sadness', 0], 'emotions_1': ['happiness', 0], 'emotions_2': ['anger', 0],
              'emotions_3': ['surprise', 0], 'emotions_4': ['fear', 0], 'emotions_5': ['disgust', 0],
              'emotions_6': ['neutral', 0],
              'pain_0': ['acute pain', 1], 'pain_1': ['chonic_pain', 1], 'sensitivity_0': ['tactile sensitivity', 1],
              'sensitivity_1': ['nociceptive sensitivity', 1], 'sensitivity_2': ['hedonic sensitivity', 1]}

color_defs = {'head': 26, 'shoulders':128, 'arms': 102, 'upper_torso': 51, 'lower_torso': 77, 'legs': 153,
              'hands': 204, 'feet': 230}

rois_base = imageio.imread(maskloc + 'kipu_ROI_new.png', as_gray=True, pilmode='L')

bg = pd.read_csv(bgdatapath)

for j, cond in enumerate(stim_names.keys()):
    with h5py.File(datafile, 'r') as h:
        data = h[cond].value
    for roi in color_defs:
        print('working on', cond, ':', roi)
        if stim_names[cond][1] == 0:
            rois = rois_base.copy()
        else:
            rois = np.hstack((rois_base, rois_base))
        rois[rois != color_defs[roi]] = 0
        rois[rois > 0] = 1
        pos_n, pos_prop, neg_n, neg_prop = count_pixels_posneg(data, rois)
        bg[cond + '_'+roi+ '_pos_color'] = pos_prop
        if stim_names[cond][1] == 0:
            bg[cond + '_'+roi+ '_neg_color'] = neg_prop
            bg[cond + '_'+roi+ '_total'] = list(map(add, pos_prop, neg_prop))

bg.to_csv(outfilename)

#
# outline_back_better = outline_back.copy()
# outline_back_better[outline_back_better <= 20] = 0
# outline_back_better[outline_back_better > 20] = 1
# outline_front_better = outline_front.copy()
# outline_front_better[outline_front_better <= 20] = 0
# outline_front_better[outline_front_better > 20] = 1
#
# for roi in color_defs:
#

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
# #
# plt.hist(kipu.flatten())
# plt.savefig(maskloc + 'hedonic_sensitivity_values.png')
# # plt.hist(outline_back)
# # plt.savefig(maskloc + 'outline_back_values.png')
#
# # Visualizing ROIs to confirm colours
# rois = rois_base.copy()
# rois[rois!=color_defs['feet']] = 0
# scipy.misc.imsave(maskloc + 'roi_test_feet.png', rois)
#
# rois = rois_base.copy()
# rois[rois!=color_defs['legs']] = 0
# scipy.misc.imsave(maskloc + 'roi_test_legs.png', rois)
#
# rois = rois_base.copy()
# rois[rois!=color_defs['crotch']] = 0
# scipy.misc.imsave(maskloc + 'roi_test_crotch.png', rois)
#
# rois = rois_base.copy()
# rois[rois!=color_defs['head']] = 0
# scipy.misc.imsave(maskloc + 'roi_test_head.png', rois)
#
# rois = rois_base.copy()
# rois[rois!=color_defs['arms']] = 0
# scipy.misc.imsave(maskloc + 'roi_test_arms.png', rois)
#
# rois = rois_base.copy()
# rois[rois!=color_defs['hands']] = 0
# scipy.misc.imsave(maskloc + 'roi_test_hands.png', rois)
#
# rois = rois_base.copy()
# rois[rois!=color_defs['torso']] = 0
# scipy.misc.imsave(maskloc + 'roi_test_torso.png', rois)