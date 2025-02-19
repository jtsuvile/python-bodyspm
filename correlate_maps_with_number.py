from bodyfunctions import *
import h5py
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.stats import pointbiserialr


import sys

figloc = '/Users/juusu53/Documents/projects/endometrioosi/figures/'
maskloc = '/Users/juusu53/Documents/projects/kipupotilaat/python_code/sample_data/'
dataloc = '/Volumes/Shield1/kipupotilaat/data/endometriosis/processed/'
datafile = get_latest_datafile(dataloc)

dataloc_controls = '/Volumes/Shield1/kipupotilaat/data/endometriosis/processed_other_mixed/'
datafile_controls = get_latest_datafile(dataloc_controls)

mask_fb = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')
mask_one = read_in_mask(maskloc + 'mask_front_new.png')

stim_names = {
              'pain_0': ['current pain', 1], 'pain_1': ['chonic pain', 1], 
              }



cond = "pain_0"
corr_variable_name = 'feels_anxiety'

with h5py.File(datafile, 'r') as h:
    bodymap = h[cond][()]
    corr_variable = h[corr_variable_name][()]
    # replacing numeric representation of missing values with nan
    corr_variable = [x if x >= 0 else np.nan for x in corr_variable]

# fix plot color definitions
hot = plt.cm.get_cmap('hot', 256)
new_cols = hot(np.linspace(0, 1, 256))

cold = np.hstack((np.fliplr(new_cols[:,0:3]),new_cols[:,3][:,None]))
newcolors = np.vstack((np.flipud(cold), new_cols))
newcolors = np.delete(newcolors, np.arange(200, 312, 2), 0)

mask = mask_fb
cmap = 'coolwarm'
#cmap = ListedColormap(newcolors)
vmin = -1
vmax = 1


bodymap_bin = binarize(bodymap.copy())
dim1 = bodymap_bin.shape[1]
dim2 = bodymap_bin.shape[2]

result_map_r = np.zeros([dim1, dim2])
result_map_p = np.zeros([dim1, dim2])

# loop over all pixels and count correlation
for ind_i in range(0,dim1):
    for ind_j in range(0,dim2):
        # handle NaNs

        # point biserial correlation does not like NaNs so we need to handle those
        problem_indices_bodymap = np.argwhere(np.isnan(bodymap_bin[:, ind_i, ind_j]))
        problem_indices_corrvariable = np.argwhere(np.isnan(corr_variable))
        all_problem_indices = list(set(problem_indices_corrvariable.flatten()) | set(problem_indices_bodymap.flatten()))

        # drop problematic values
        bodymap_bin_no_nans = np.delete(bodymap_bin[:, ind_i, ind_j], all_problem_indices) 
        corr_variable_no_nans = np.delete(corr_variable, all_problem_indices) 

        if len(set(bodymap_bin_no_nans))==1 | len(set(corr_variable_no_nans)) == 1:
            correlation = 0
            p_value = 1
        else:
            correlation, p_value = pointbiserialr(bodymap_bin_no_nans, corr_variable_no_nans)
        result_map_p[ind_i, ind_j] = p_value
        result_map_r[ind_i, ind_j] = correlation

result_map_r_no_fdr = result_map_r.copy()
result_map_r_with_fdr = result_map_r.copy()

result_map_r_no_fdr[result_map_p > 0.05] = 0
masked_result_map_no_fdr = np.ma.masked_where(mask != 1, result_map_r_no_fdr)

result_map_p_corrected, twosamp_reject = p_adj_maps(result_map_p, mask=mask, method='fdr_bh')
result_map_r_with_fdr[result_map_p_corrected > 0.05] = 0
result_map_r_with_fdr[np.isnan(result_map_p_corrected)] = 0

# this line does not do what it should!
masked_result_map_with_fdr = np.ma.masked_where(mask != 1, result_map_r_with_fdr)

unique, counts = np.unique(result_map_r_with_fdr, return_counts=True)
foo = pd.DataFrame.from_dict(dict(zip(unique, counts)), orient='index').reset_index().sort_values(0, ascending = False)

fig = plt.figure(figsize=(20, 10))

ax1 = plt.subplot(131)
img1 = plt.imshow(result_map_r, cmap=cmap, vmin=vmin, vmax=vmax)
ax1.title.set_text(f"raw correlation, no cleanup")
fig.colorbar(img1, fraction=0.046, pad=0.04)
ax1.axis('off')

ax2 = plt.subplot(132)
img2 = plt.imshow(masked_result_map_no_fdr, cmap=cmap, vmin=-1, vmax=1)
ax2.title.set_text('Correlations no FDR')
fig.colorbar(img2, fraction=0.046, pad=0.04)
ax2.axis('off')

ax3 = plt.subplot(133)
img3 = plt.imshow(masked_result_map_with_fdr, cmap=cmap, vmin=-1, vmax=1)
ax3.title.set_text('Correlations with FDR')
fig.colorbar(img3, fraction=0.046, pad=0.04)
ax3.axis('off')

plt.savefig(figloc+f'corr_maps_{cond}_vs_{corr_variable_name}.png')
plt.close()
