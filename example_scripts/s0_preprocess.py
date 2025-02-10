import os
import sys
import pandas as pd
from bodyspm.classdefinitions import Subject, Stimuli
from bodyspm.bodyfunctions import combine_data, preprocess_subjects
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# set up stimuli description
onesided = [True, True, True, True, True, True, True, False, False, False]
# boolean or list of booleans describing if data is onesided (e.g. emotion body maps, with one image
# representing intensifying and one image representing lessening activation. In this case, one side is deducted from
# the other. Alternative (False) describes situation where both sides of colouring are retained, e.g. touch allowances
# for front and back of body.
data_names = ['emotions_0', 'emotions_1', 'emotions_2', 'emotions_3', 'emotions_4','emotions_5','emotions_6', 'sensitivity_0','sensitivity_1','sensitivity_2']
stim_names = ['stim1','stim2','stim3','stim4','stim5', 'pain1', 'pain2'] # potentially add stimulus names for more intuitive data handling

# inputs
repo_root = Path(__file__).resolve().parent.parent
dataloc = repo_root / "sample_data" 
outdataloc = repo_root / "processed"
Path(outdataloc).mkdir(parents=True, exist_ok=True)

# what file to use for output
subject_info_csv_filename = outdataloc / "subject_background.csv"

#dataloc = '/home/bodymaps/subjects/'
#outdataloc = '/home/bodymaps/processed/'
#csvname = '/home/bodymaps/subject_background.csv'

subnums = ['test_sub_1', 'test_sub_2', 'test_sub_3','test_sub_4']
bg_files = ['data.txt']
fieldnames = [['sex','age','height','weight','handedness','education','physical_work','sitting_work','profession','history_of_x','history_of_y','history_of_z']]

# define stimulus set
stim = Stimuli(data_names, onesided=onesided)

# read subjects from web output and write out to a more sensible format
preprocess_subjects(subnums, str(dataloc), str(outdataloc), stim, bg_files, fieldnames)

# Gather subjects into one dict
print("combining data from ", len(subnums), " subjects")
print("getting started")
subnums = ['test_sub_1', 'test_sub_2', 'test_sub_3','test_sub_4']
grouping = ['foo', 'bar', 'foo', 'bar']

## TODO: make a smarter system to handle subject IDs that are not strictly int, hdf5 does not want them
full_dataset = combine_data(str(outdataloc), subnums, save=True, noImages=False)

bg = full_dataset['bg']
bg.to_csv(subject_info_csv_filename)
