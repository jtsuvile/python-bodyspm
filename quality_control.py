import os
import sys
import pandas as pd
from classdefinitions import Subject, Stimuli
from bodyfunctions import make_qc_figures, preprocess_subjects, intentionally_empty
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import math


who = 'helsinki'

if who == 'control':
    dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/subjects/'
    outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/qc/'
    subfile = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/subs.txt'
    csvname = '/m/nbe/scratch/socbrain/kipupotilaat/data/bg_all_controls.csv'
elif who == 'helsinki':
    dataloc = '/Users/juusu53/Documents/projects/kipupotilaat/data/subjects/'
    outdataloc = '/Users/juusu53/Documents/projects/kipupotilaat/data/qc/helsinki/'
    outdataloc2 = '/Users/juusu53/Documents/projects/kipupotilaat/data/processed/'
    #dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/subjects/'
    #outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/qc/'
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
display_names = ['sadness', 'happiness', 'anger', 'surprise', 'fear', 'disgust', 'neutral',
'current pain', 'chonic pain', 'tactile sensitivity','nociceptive sensitivity', 'hedonic sensitivity']
stim = Stimuli(data_names, onesided=onesided, show_names = display_names)

# if who != 'matched_controls_helsinki' and who != 'matched_controls_two_each':
#     print('re-reading')
#     with open(subfile) as f:
#         subnums = f.readlines()
#     subnums = [x.strip() for x in subnums]
#
# subnum='3217'
# preprocess_subjects([subnum], dataloc, outdataloc2, stim)
# sub = Subject(subnum)
# sub.read_data(dataloc, stim, whole_image=False)
#
# #make_qc_figures(subnums, dataloc, outdataloc, stim)
# # NB: square for marking intentionally empty bodies approximately at
# # [530:580,430:480] in the full image
#
# fileloc = outdataloc
# # make sure non coloured values are white in twosided datas
# twosided_cmap = plt.get_cmap('Greens')
# twosided_cmap.set_under('white', 1.0)
# twosided_cmap.set_bad('grey',1.0)
# fig, axes = plt.subplots(figsize=(24, 10), ncols=math.ceil(len(sub.data.keys())/2), nrows=2)
# for i, (key, value) in enumerate(sub.data.items()):
#     if i < len(sub.data.items())/2:
#         row = 0
#         col = i
#     else:
#         row = 1
#         col = i - math.floor(len(sub.data.items())/2)
#     print(np.count_nonzero(value[530:580,430:480]))
#     map_to_plot = value
#     map_to_plot_2 = np.ma.masked_where(np.isnan(map_to_plot),map_to_plot)
#     if intentionally_empty(map_to_plot):
#         map_to_plot[530:580,430:480] = 0.1
#     img = axes[row,col].imshow(map_to_plot_2, cmap=twosided_cmap, vmin=np.finfo(float).eps, vmax=0.05)
#     fig.colorbar(img, ax=axes[row,col], fraction=0.04, pad=0.04)
#     if 'show_name' in stim.all[key]:
#         axes[row, col].set_title(stim.all[key]['show_name'])
#     else:
#         axes[row, col].set_title(key)
# fig.suptitle("subject : " + sub.name)
# fig.tight_layout()
# fileloc_fig = fileloc + '/figures/'
# if not os.path.exists(fileloc_fig):
#     os.makedirs(fileloc_fig)
# filename = fileloc_fig+ '/sub_' + str(sub.name) + '_qc_test.png'
# #plt.show()
# plt.savefig(filename, bbox_inches='tight')
#
# sub.draw_sub_data(Stimuli(data_names[0:8], onesided[0:8]), fileloc, qc=True)