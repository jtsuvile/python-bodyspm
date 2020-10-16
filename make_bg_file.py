from bodyfunctions import *
import pandas as pd

dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/stockholm/processed/'
datafile = get_latest_datafile(dataloc)

bg_variables = ['subid', 'sex', 'age', 'weight', 'height', 'handedness', 'education', 'work_physical', 'work_sitting',
                'psychologist',	'psychiatrist', 'neurologist', 'pain_now', 'pain_last_day', 'pain_chronic',
                'hist_migraine', 'hist_headache', 'hist_abdomen', 'hist_back_shoulder',	'hist_joint_limb',
                'hist_menstrual', 'painkillers_overcounter', 'painkillers_prescription', 'painkillers_othercns',
                'feels_pain', 'feels_depression', 'feels_anxiety', 'feels_happy', 'feels_sad', 'feels_angry',
                'feels_fear', 'feels_surprise',	'feels_disgust', 'bpi_worst', 'bpi_least', 'bpi_average',
                'bpi_now', 'bpi_painkiller_relief',	'bpi_functioning', 'bpi_mood', 'bpi_walk',	'bpi_work',
                'bpi_relationships', 'bpi_sleep', 'bpi_enjoyment']

bg = pd.DataFrame()
with h5py.File(datafile, 'r') as fid:
    for i, bgname in enumerate(bg_variables):
        bg[bgname] = fid[bgname].value

bg.to_csv('/m/nbe/scratch/socbrain/kipupotilaat/data/bg_from_h5py_stockholm.csv', na_rep='NaN')