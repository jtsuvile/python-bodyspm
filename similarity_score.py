from bodyfunctions import *
import h5py
import numpy as np
from itertools import combinations
from scipy.spatial.distance import jaccard, hamming
from scipy.stats import spearmanr

who = 'controls'
distance_metric = 'spearman'

if who == 'controls':
    dataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/controls/all/'
    datafile_controls = get_latest_datafile(dataloc)
    with h5py.File(datafile_controls, 'r') as c:
        data = c['emotions_0'][()]
        n_subs = len(data)
    threshold = 0.007

elif who == 'patients':
    dataloc1 = '/Volumes/Shield1/kipupotilaat/data/stockholm/processed/all/'
    datafile_patients = get_latest_datafile(dataloc1)
    with h5py.File(datafile_patients, 'r') as c:
        data_patients = c['emotions_0'][()]
        n_subs = len(data_patients)
    threshold = 0.001
    

outfilename = f'/Volumes/Shield1/kipupotilaat/data/stockholm/{distance_metric}_distance_emotions_{who}.csv'

maskloc = '/Users/juusu53/Documents/projects/kipupotilaat/python_code/sample_data/'

stim_names = {
    'emotions_0': ['Sadness', 0],
    'emotions_2': ['Anger', 0],
    'emotions_3': ['Surprise', 0],
    'emotions_4': ['Fear', 0],  
    'emotions_5': ['Disgust', 0],
    'emotions_1': ['Happiness', 0]}

stimuli = list(stim_names.keys())

res = pd.DataFrame(np.nan, columns=[f"{stim_names[stimuli[0]][0]}-{stim_names[stimuli[1]][0]}",
                            f"{stim_names[stimuli[0]][0]}-{stim_names[stimuli[2]][0]}",
                            f"{stim_names[stimuli[0]][0]}-{stim_names[stimuli[3]][0]}",
                            f"{stim_names[stimuli[0]][0]}-{stim_names[stimuli[4]][0]}",
                            ],
                            index = range(0,n_subs))

for cond1_name, cond2_name in combinations(stimuli, 2):
    print(f"working on {cond1_name} + {cond2_name}")

    if who=='controls':
        with h5py.File(datafile_controls, 'r') as c:
            cond1 = c[cond1_name][()]
            cond2 = c[cond2_name][()]
    elif who=='patients':
        with h5py.File(datafile_patients, 'r') as c:
            cond1 = c[cond1_name][()]
            cond2 = c[cond2_name][()]

    cond1 = binarize(cond1, threshold=threshold)
    cond2 = binarize(cond2, threshold=threshold)

    for i in range(n_subs):
        curr_subject_cond_1 = np.concatenate(cond1[i])
        curr_subject_cond_2 = np.concatenate(cond2[i])
        if distance_metric == 'jaccard':
            curr_subject_res = jaccard(curr_subject_cond_1, curr_subject_cond_2)
        elif distance_metric == 'hamming':
            curr_subject_res = hamming(curr_subject_cond_1, curr_subject_cond_2)
        elif distance_metric == 'spearman':
            curr_subject_res, curr_subject_p = spearmanr(curr_subject_cond_1, curr_subject_cond_2)
        res.loc[i, f"{stim_names[cond1_name][0]}-{stim_names[cond2_name][0]}"] = curr_subject_res

res.to_csv(outfilename, index = False)
