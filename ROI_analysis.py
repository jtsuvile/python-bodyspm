from bodyfunctions import *
import h5py
import numpy as np
import imageio
from operator import add


bgdatapath = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/endometriosis_patients_18_03_2022.csv'
dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/processed/'
outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/endometriosis_patients_with_activations_by_ROI_03_2022.csv'

# bgdatapath = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/endometriosis_controls_18_03_2022.csv'
# dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/matched_controls/'
# outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/endometriosis_controls_with_activations_by_ROI_03_2022.csv'

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

