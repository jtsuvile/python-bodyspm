from bodyfunctions import *
import h5py
import numpy as np
from itertools import combinations
from scipy.spatial.distance import jaccard, hamming

who = 'patients'
distance_metric = 'jaccard'

if who == 'controls':
    dataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/controls/all/'
    datafile_controls = get_latest_datafile(dataloc)
    with h5py.File(datafile_controls, 'r') as c:
        data = c['emotions_0'][()]
        n_subs = len(data)
    threshold = 0.007

elif who == 'patients':
    dataloc1 = '/Volumes/Shield1/kipupotilaat/data/stockholm/processed/fibro/'
    dataloc2 = '/Volumes/Shield1/kipupotilaat/data/stockholm/processed/clbp/'
    datafile_fibro = get_latest_datafile(dataloc1)
    datafile_lbp = get_latest_datafile(dataloc2)
    with h5py.File(datafile_fibro, 'r') as c:
        data_fibro = c['emotions_0'][()]
        n_subs_fibro = len(data_fibro)
    with h5py.File(datafile_lbp, 'r') as c:
        data_lbp = c['emotions_0'][()]
        n_subs_lbp = len(data_lbp)
    n_subs = n_subs_fibro+n_subs_lbp
    threshold = 0.001
    

outfilename = f'/Volumes/Shield1/kipupotilaat/data/stockholm/{distance_metric}_distance_emotions_{who}.csv'

maskloc = '/Users/juusu53/Documents/projects/kipupotilaat/python_code/sample_data/'

stim_names = {
    'emotions_0': ['Sadness', 0],
    'emotions_2': ['Anger', 0],
    'emotions_3': ['Surprise', 0],
    'emotions_4': ['Fear', 0],  
    'emotions_5': ['Disgust', 0]}

stimuli = list(stim_names.keys())

res = pd.DataFrame(np.nan, columns=[f"{stimuli[0]}-{stimuli[1]}",
                            f"{stimuli[0]}-{stimuli[2]}",
                            f"{stimuli[0]}-{stimuli[3]}",
                            f"{stimuli[0]}-{stimuli[4]}",
                            ],
                            index = range(0,n_subs))

for cond1_name, cond2_name in combinations(stimuli, 2):
    print(f"working on {cond1_name} + {cond2_name}")

    if who=='controls':
        with h5py.File(datafile_controls, 'r') as c:
            cond1 = c[cond1_name][()]
            cond2 = c[cond2_name][()]
    elif who=='patients':
        #
        with h5py.File(datafile_lbp, 'r') as c:
            cond1_lbp = c[cond1_name][()]
            cond2_lbp = c[cond2_name][()]
        with h5py.File(datafile_fibro, 'r') as c:
            cond1_fibro = c[cond1_name][()]
            cond2_fibro = c[cond2_name][()]
        cond1 =  np.concatenate((cond1_lbp, cond1_fibro))
        cond2 =  np.concatenate((cond2_lbp, cond2_fibro))

    cond1 = binarize(cond1, threshold=threshold)
    cond2 = binarize(cond2, threshold=threshold)

    for i in range(n_subs):
        curr_subject_cond_1 = np.concatenate(cond1[i])
        curr_subject_cond_2 = np.concatenate(cond2[i])
        if distance_metric == 'jaccard':
            curr_subject_res = jaccard(curr_subject_cond_1, curr_subject_cond_2)
        elif distance_metric == 'hamming':
            curr_subject_res = hamming(curr_subject_cond_1, curr_subject_cond_2)
        res.loc[i, f"{cond1_name}-{cond2_name}"] = curr_subject_res

res.dropna().to_csv(outfilename, index = False)
