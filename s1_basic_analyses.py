from classdefinitions import Subject, Stimuli
from bodyfunctions import compare_groups, correlate_maps
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

# Read pickled dataset from file

dataloc = '/Users/jtsuvile/Documents/projects/kipupotilaat/python_code_testing/'
datafile = dataloc + 'full_dataset.pickle'
all_data = pickle.load(open(datafile, "rb" ))

alpha = 0.2 # for testing with the tiny data set, in real use you would set this to 0.05/0.01 or whatever alpha level you choose

# one sample t-test
statistics, pval = stats.ttest_1samp(all_data['emotions_1'], 0, nan_policy = 'omit', axis=0)

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

g0_indices = [i for i, x in enumerate(groupdefinitions) if x == groups_to_compare[0]]
g1_indices = [i for i, x in enumerate(groupdefinitions) if x == groups_to_compare[1]]
which_stim = 'sensitivity_0'

comparison_stats, comparison_p = compare_groups(all_data[which_stim], g0_indices, g1_indices)

# average maps (as proportion)
mapname = 'sensitivity_0'
map = all_data[mapname]
onesided = all_data['stimuli'].all[mapname]['onesided']

if onesided:
    # one-sided
    map[map<0] = -1
    map[map>0] = 1
    propdata = np.nansum(map, axis=0) /sum(~np.isnan(map))
else:
    # two-sided
    propdata = np.nansum(np.ceil(abs(map)), axis = 0) /sum(~np.isnan(map))

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