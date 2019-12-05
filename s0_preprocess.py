import os
import sys
import pandas as pd
from classdefinitions import Subject, Stimuli
from bodyfunctions import combine_data, preprocess_subjects
import matplotlib.pyplot as plt
import numpy as np

# set up stimuli description
onesided = [True, True, True, True, True, True, True, False, False, False]
# boolean or list of booleans describing if data is onesided (e.g. emotion body maps, with one image
# representing intensifying and one image representing lessening activation. In this case, one side is deducted from
# the other. Alternative (False) describes situation where both sides of colouring are retained, e.g. touch allowances
# for front and back of body.
data_names = ['emotions_0', 'emotions_1', 'emotions_2', 'emotions_3', 'emotions_4','emotions_5','emotions_6', 'sensitivity_0','sensitivity_1','sensitivity_2']
stim_names = ['stim1','stim2','stim3','stim4','stim5', 'pain1', 'pain2'] # potentially add stimulus names for more intuitive data handling

# inputs
#dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/subjects/'
#outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/'

dataloc = '/Users/jtsuvile/Documents/projects/kipupotilaat/python_code/sample_data/'
outdataloc = '/Users/jtsuvile/Documents/projects/kipupotilaat/python_code_testing/'
subnums = ['test_sub_1', 'test_sub_2', 'test_sub_3','test_sub_4']
bg_files = ['data.txt']
fieldnames = [['sex','age','height','weight','handedness','education','physical_work','sitting_work','profession','history_of_x','history_of_y','history_of_z']]

# define stimulus set
stim = Stimuli(data_names, onesided=onesided)

# read subjects from web output and write out to a more sensible format
preprocess_subjects(subnums, dataloc, outdataloc, stim, bg_files, fieldnames)

# Gather subjects into one dict
# subnums = ['test_sub_1', 'test_sub_2', 'test_sub_3','test_sub_4']
# outdataloc = '/Users/jtsuvile/Documents/projects/kipupotilaat/python_code_testing/'
# grouping = ['foo', 'bar', 'foo', 'bar']



full_dataset = combine_data(outdataloc, subnums, groups = grouping, save=False, noImages=True)

indices = lambda searchList, elem: [[i for i, x in enumerate(searchList) if x == e] for e in elem]
indices(full_dataset['groups'], ['foo','bar'])

# # test reading subject from file
#
# # stim2 = Stimuli(fileloc='/Users/jtsuvile/Documents/projects/kipupotilaat/python_code_testing/', from_file=True)
# # sub2 = Subject('375451')
# # sub2.read_sub_from_file('/Users/jtsuvile/Documents/projects/kipupotilaat/python_code_testing/')
#
#
# ## code snippets for potential later use
#
# # np.sum(all_res['emotions_0'], axis=0).shape
#
# # NB [outdata[key]['name'] for key in outdata.keys()]
#
# #for key, value in sub.data.items():
# #    print(value['rawdata'])
