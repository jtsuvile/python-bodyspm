from bodyspm.classdefinitions import Subject, Stimuli
from bodyspm.bodyfunctions import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
from scipy import stats
import pandas as pd

# Read pickled dataset from file

dataloc = '/Users/jtsuvile/Documents/projects/kipupotilaat/python_code_testing/'
datafile = dataloc + 'full_dataset.pickle'
all_data = pickle.load(open(datafile, "rb" ))

mask_use = read_in_mask(dataloc + 'mask_front_new.png',dataloc + 'mask_back_new.png')

# fig = plt.figure()
# img = plt.imshow(mask_use)
# fig.colorbar(img)
# plt.show()

alpha = 0.2 # for testing with the tiny data set, in real use you would set this to 0.05/0.01 or whatever alpha level you choose

# one sample t-test
statistics, pval = one_sample_t_test(all_data['sensitivity_0'])
pval_adjusted = p_adj_maps(pval, mask_use)

lim = max(np.nanmax(statistics), abs(np.nanmin(statistics)))
stats_to_show = statistics.copy()
pval[np.isnan(pval_adjusted)] = 1
stats_to_show[pval_adjusted>alpha] = 0

fig = plt.figure()
img = plt.imshow(stats_to_show, cmap='RdBu_r', vmin=-10, vmax=10)
fig.colorbar(img)
plt.show()

fig = plt.figure()
img = plt.imshow(mask_use)
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

fig = plt.figure()
img = plt.imshow(comparison_stats, cmap='RdBu_r', vmin=-10, vmax=10)
fig.colorbar(img)
plt.show()

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

##
# counts
##

mapname = 'sensitivity_0'
this_map = all_data[mapname]

count_pixels(this_map)
count_pixels(this_map, mask_use)

## NB: maybe not indexing properly?
mask_indices = np.nonzero(mask_use)
inside_mask = this_map[:,mask_indices[0], mask_indices[1]]
this_map_copy = np.copy(this_map)
this_map_copy[:, mask_indices[0],mask_indices[1]] = inside_mask


fig = plt.figure()
img = plt.imshow(this_map_copy[3,:,:])
fig.colorbar(img)
plt.show()