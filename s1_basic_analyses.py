from classdefinitions import Subject, Stimuli
from bodyfunctions import combine_data
import pickle
from scipy import stats
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
groups_to_compare = ('foo', 'bar')
groupdefinitions = ['foo', 'foo', 'bar', 'bar']  # this will be later integrated to subject info & pickle
which_stim = 'sensitivity_0'

g0_indices = [i for i,x in enumerate(groupdefinitions) if x==groups_to_compare[0]]
g1_indices = [i for i,x in enumerate(groupdefinitions) if x==groups_to_compare[1]]

# copy the data for each group to avoid accidentally making edits to original data
g0_data = np.copy(all_data[which_stim][g0_indices])
g1_data = np.copy(all_data[which_stim][g1_indices])

# test of proportions
g0_data[g0_data>0] = 1
g0_data[g0_data<0] = -1
g1_data[g1_data>0] = 1
g1_data[g1_data<0] = -1

successes = [np.concatenate(np.sum(g0_data, axis=0)), np.concatenate(np.sum(g1_data, axis=0))]
counts = [np.concatenate(np.nansum(~np.isnan(g0_data), axis=0)), np.concatenate(np.nansum(~np.isnan(g1_data), axis=0))]
# not ready
# proportions_ztest(successes, counts)

# two sample t-test
statistics_twosamp, pval_twosamp = stats.ttest_ind(g0_data, g1_data, axis=0, nan_policy='omit')

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


# plot
twosided_cmap = plt.get_cmap('Greens')
twosided_cmap.set_under('white', 1.0)

fig = plt.figure()
if onesided:
    img = plt.imshow(propdata, cmap='RdBu_r', vmin=-1, vmax=1)
else:
    img = plt.imshow(propdata, cmap=twosided_cmap, vmin=0, vmax=1)

fig.colorbar(img)
plt.show()

# correlations with X
which_stim = 'sensitivity_0'
data = all_data[which_stim]
dims = data.shape
feature_to_corr_with = [1, 2, 3, 4]


data_reshaped = np.reshape(data, (dims[0], -1))

result = map(lambda x: np.correlate(x, feature_to_corr_with), data_reshaped)

data_re_reshaped = np.reshape(data_reshaped, dims)
