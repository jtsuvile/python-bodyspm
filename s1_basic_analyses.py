from classdefinitions import Subject, Stimuli
from bodyfunctions import compare_groups, correlate_maps, one_sample_t_test
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import pandas as pd

# Read pickled dataset from file

dataloc = '/Users/jtsuvile/Documents/projects/kipupotilaat/python_code_testing/'
datafile = dataloc + 'full_dataset.pickle'
all_data = pickle.load(open(datafile, "rb" ))

alpha = 0.2 # for testing with the tiny data set, in real use you would set this to 0.05/0.01 or whatever alpha level you choose

# one sample t-test
statistics, pval = one_sample_t_test(all_data['sensitivity_0'])

lim = max(np.nanmax(statistics), abs(np.nanmin(statistics)))

stats_to_show = statistics.copy()
pval[np.isnan(pval)] = 1
stats_to_show[pval>alpha] = 0

fig = plt.figure()
img = plt.imshow(stats_to_show, cmap='RdBu_r', vmin=-10, vmax=10)
fig.colorbar(img)
plt.show()

#
# compare two groups
#
groups_to_compare = ['foo', 'bar']
groupdefinitions = ['foo', 'foo', 'bar', 'bar']  # this will be later integrated to subject info & pickle

group1 = [i for i, x in enumerate(groupdefinitions) if x == groups_to_compare[0]]
group2 = [i for i, x in enumerate(groupdefinitions) if x == groups_to_compare[1]]
which_stim = 'sensitivity_0'

comparison_stats, comparison_p = compare_groups(all_data[which_stim], group1, group2, testtype='z')

# average maps (as proportion)
mapname = 'sensitivity_0'
this_map = all_data[mapname]
onesided = all_data['stimuli'].all[mapname]['onesided']

if onesided:
    # one-sided
    this_map[this_map<0] = -1
    this_map[this_map>0] = 1
    propdata = np.nansum(this_map, axis=0) /sum(~np.isnan(this_map))
else:
    # two-sided
    propdata = np.nansum(np.ceil(abs(this_map)), axis = 0) /sum(~np.isnan(this_map))

# correlations with X
which_stim = 'sensitivity_0'
data = all_data[which_stim]
corr_with = all_data['bg']['sitting_work'].astype(float)
corr_map = correlate_maps(data, corr_with)

# test success with plotting
twosided_cmap = plt.get_cmap('Greens')
twosided_cmap.set_under('white', 1.0)
onesided = True
showdata = corr_map

fig = plt.figure()
if onesided:
    img = plt.imshow(showdata, cmap='RdBu_r', vmin=-1, vmax=1)
else:
    img = plt.imshow(showdata, cmap=twosided_cmap, vmin=0, vmax=1)

fig.colorbar(img)
plt.show()