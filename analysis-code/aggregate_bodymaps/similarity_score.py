import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir) 

from bodyfunctions import *
import h5py
import numpy as np
from itertools import combinations
from scipy.spatial.distance import jaccard, hamming
from scipy.stats import spearmanr

# settings
who = 'patients'
what = 'twosided'
distance_metric = 'jaccard'
outfilename = f'/Volumes/Shield1/kipupotilaat/data/stockholm/intermediate/{distance_metric}_distance_{what}_{who}.csv'
maskloc = '/Users/juusu53/Documents/projects/kipupotilaat/python_code/sample_data/'

if what == 'emotions':
    mask = read_in_mask(maskloc + 'mask_front_new.png')
    stim_names = {
        'emotions_0': ['Sadness', 0],
        'emotions_2': ['Anger', 0],
        'emotions_3': ['Surprise', 0],
        'emotions_4': ['Fear', 0],  
        'emotions_5': ['Disgust', 0],
        'emotions_1': ['Happiness', 0],
        'emotions_6': ['Neutral', 0]}
if what == 'twosided':
    mask = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')
    stim_names = {'pain_0': ['Current pain', 1],
                'pain_1': ['Chonic pain', 1],
                'sensitivity_0': ['Tactile sensitivity', 1],
                'sensitivity_1': ['Nociceptive sensitivity', 1],
                'sensitivity_2': ['Hedonic sensitivity', 1]}

stimuli = list(stim_names.keys())

# 
if who == 'controls':
    dataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/controls/all/'
    datafile = get_latest_datafile(dataloc)
    threshold = 0.007

elif who == 'patients':
    dataloc1 = '/Volumes/Shield1/kipupotilaat/data/stockholm/processed/all/'
    datafile = get_latest_datafile(dataloc1)
    threshold = 0.001
    
with h5py.File(datafile, 'r') as c:
    subs = c['subid'][()]
    n_subs = len(subs)



res = pd.DataFrame(np.nan, columns=[
    f"{distance_metric}_{stim_names[stimuli[0]][0]}_{stim_names[stimuli[1]][0]}",
                            f"{distance_metric}_{stim_names[stimuli[0]][0]}_{stim_names[stimuli[2]][0]}",
                            f"{distance_metric}_{stim_names[stimuli[0]][0]}_{stim_names[stimuli[3]][0]}",
                            f"{distance_metric}_{stim_names[stimuli[0]][0]}_{stim_names[stimuli[4]][0]}",
                            ],
                            index = subs)

for cond1_name, cond2_name in combinations(stimuli, 2):
    print(f"working on {cond1_name} + {cond2_name}")

    with h5py.File(datafile, 'r') as c:
        cond1 = c[cond1_name][()]
        cond2 = c[cond2_name][()]

    cond1 = binarize(cond1, threshold=threshold)
    cond2 = binarize(cond2, threshold=threshold)

    for i, sub in enumerate(subs):
        curr_subject_cond_1 = extract_masked_vector(cond1[i], mask)
        curr_subject_cond_2 = extract_masked_vector(cond2[i], mask)
        if distance_metric == 'jaccard':
            curr_subject_res = jaccard(curr_subject_cond_1, curr_subject_cond_2)
        elif distance_metric == 'hamming':
            curr_subject_res = hamming(curr_subject_cond_1, curr_subject_cond_2)
        elif distance_metric == 'spearman':
            curr_subject_res, curr_subject_p = spearmanr(curr_subject_cond_1, curr_subject_cond_2)
        res.loc[sub, f"{distance_metric}_{stim_names[cond1_name][0]}_{stim_names[cond2_name][0]}"] = curr_subject_res

res.reset_index().rename(columns={"index":"subid"}).to_csv(outfilename, index = False)
