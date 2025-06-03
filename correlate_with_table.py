import pandas as pd
from bodyfunctions import make_correlation_plot, binarize, correlate_maps
from bodyfunctions import get_latest_datafile, read_in_mask, align_data
import h5py

dataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/processed/all'
tableloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/intermediate/' +\
    'stockholm_questionnaires_scored.csv'
outdataloc = '/Users/juusu53/Documents/projects/kipupotilaat/stockholm/r_code/figures/'
interesting_variables = ["styrka_nu", "styrka_genomsnitt",
                         "lidande_nu", "lidande_genomsnitt",
                         "obehag_nu", "obehag_genomsnitt",
                         "pain_duration", "score_pain_intensity", "score_bpi_inference"]
# "score_bdi", "score_stai_state", "score_stai_trait"]
which_maps = ['pain_0', 'pain_1',
              'sensitivity_0', 'sensitivity_1', 'sensitivity_2']
analysis = 'pointbiserialr'
# which_maps = ['emotions_0', 'emotions_1', 'emotions_2', 'emotions_3',
#               'emotions_4', 'emotions_5', 'emotions_6']
# analysis = 'spearmanr'

threshold = 0.001  # 0.001 for patients, 0.007 for controls

maskloc = '/Users/juusu53/Documents/projects/kipupotilaat/' +\
    'python_code/sample_data/'
datafile = get_latest_datafile(dataloc)

mask_one = read_in_mask(maskloc + 'mask_front_new.png')
mask_fb = read_in_mask(maskloc + 'mask_front_new.png',
                       maskloc + 'mask_back_new.png')

stim_names = {'emotions_0': ['sadness', 0],
              'emotions_1': ['happiness', 0],
              'emotions_2': ['anger', 0],
              'emotions_3': ['surprise', 0],
              'emotions_4': ['fear', 0],
              'emotions_5': ['disgust', 0],
              'emotions_6': ['neutral', 0],
              'pain_0': ['current_pain', 1],
              'pain_1': ['chonic_pain', 1],
              'sensitivity_0': ['tactile_sensitivity', 1],
              'sensitivity_1': ['nociceptive_sensitivity', 1],
              'sensitivity_2': ['hedonic_sensitivity', 1]}


# read in data
df = pd.read_csv(tableloc)

for which_map in which_maps:
    with h5py.File(datafile, 'r') as h:
        subs = h['subid'][()]
        fig_data = h[which_map][()]

    bodymap, aligned_df = align_data(subs, fig_data, df, 'internetnummer')

    for variablename in interesting_variables:
        print(f"{which_map} x {variablename}")
        corr_variable = list(aligned_df[variablename])
        bodymap_bin = binarize(bodymap.copy(), threshold=threshold)

        # run analysis
        result_map_r, result_map_p = correlate_maps(bodymap_bin,
                                                    corr_variable,
                                                    method=analysis)

        # make plots
        # which mask alternative do we use for plotting?
        if stim_names[which_map][1] == 0:
            mask = mask_one
        else:
            mask = mask_fb
        suptitle = f'{analysis} correlation between {variablename} ' +\
                   f'and {stim_names[which_map][0]}'
        make_correlation_plot(result_map_r, result_map_p,
                              mask, suptitle, outdataloc)
