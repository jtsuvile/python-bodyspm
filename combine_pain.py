import os
import pandas as pd
from classdefinitions import Subject, Stimuli
import matplotlib.pyplot as plt
import numpy as np
import time
import csv

outdataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/stockholm/processed/'
grouping_file = '/m/nbe/scratch/socbrain/kipupotilaat/data/stockholm/diagnoses_KI_12_2019_no_empty_cells.csv'
group_colnames = ['date_from_KI', 'sex_from_KI', 'diagnosis', 'n', 'subid', 'dob_from_KI']

with open(grouping_file, newline='', encoding='utf-8-sig') as csvfile:
    grouping_data = list(csv.reader(csvfile, delimiter=';'))
group_df = pd.DataFrame(grouping_data)
group_df.columns = group_colnames
# cleaning up fields filled in by humans
group_df['diagnosis'] = group_df['diagnosis'].str.upper()
group_df['diagnosis'] = group_df['diagnosis'].str.replace(' ', '_')
group_df['subid'] = group_df['subid'].str.replace(' ', '')
group_df['subid'] = group_df['subid'].astype(int)

new_bgdata = group_df
linking_col = 'subid'
subloc = outdataloc
override=False

existing_subs = os.listdir(subloc)
group_colnames = new_bgdata.columns.tolist()
if linking_col in group_colnames:
    group_colnames.remove(linking_col)
else:
    print("cannot find linking identifier ", linking_col)
for subject in new_bgdata[linking_col]:
    print(subject)
    if str(subject) in existing_subs:
        temp_sub = Subject(subject)
        temp_sub.read_sub_from_file(subloc, noImages=True)
        for column in group_colnames:
            if column not in temp_sub.has_background() or override == True:
                temp_sub.add_background(column,
                                        new_bgdata.loc[new_bgdata[linking_col] == subject][column].values[0])
                temp_sub.write_sub_to_file(subloc)
            else:
                print("not updating subject ", subject, " for ", column)
    else:
        print('no subject ', subject, 'found')
print('done all')