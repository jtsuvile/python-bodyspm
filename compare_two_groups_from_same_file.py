from bodyfunctions import *
import h5py
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
from PIL import Image
from operator import add
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# define how we want to separate out our groups
group_key = 'sex'
group_1_definition = 0
group_1_name = 'men'
group_2_definition = 1
group_2_name = 'women'

# filenames
dataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/controls/test/'
outfilename = f'/Users/juusu53/Documents/projects/kipupotilaat/stockholm/figures/compare_{group_key}.png'
maskloc = '/Users/juusu53/Documents/projects/kipupotilaat/python_code/sample_data/'


datafile = get_latest_datafile(dataloc)


stim_names = {'emotions_2': ['Anger', 0],'emotions_4': ['Fear', 0],  'emotions_5': ['Disgust', 0],
              'emotions_1': ['Happiness', 0], 'emotions_0': ['Sadness', 0],
              'emotions_3': ['Surprise', 0], 'emotions_6': ['Neutral', 0]}

mask_one = read_in_mask(maskloc + 'mask_front_new.png')

# colormap for emotions
hot = plt.cm.get_cmap('hot', 256)
new_cols = hot(np.linspace(0, 1, 256))

cold = np.hstack((np.fliplr(new_cols[:,0:3]),new_cols[:,3][:,None]))
newcolors = np.vstack((np.flipud(cold), new_cols))
newcolors = np.delete(newcolors, np.arange(200, 312, 2), 0)
cmap = ListedColormap(newcolors)

# colormap for difference
hotcool = cm.get_cmap('bwr', 256)
newcolors = hotcool(np.linspace(0, 1, 256))
outlinecolor = np.array([100/256, 100/256, 100/256, 1])
newcolors = np.vstack((outlinecolor, newcolors))
newcmp = ListedColormap(newcolors)

vmin = -1
vmax = 1

#fig, axs = plt.subplots(3,7, figsize=(25, 17), facecolor='w', edgecolor='k')
#axs = axs.ravel()

fig = plt.figure(figsize=(18, 19))
suptitle = 'Average emotions'

all_figs = np.zeros([mask_one.shape[0], mask_one.shape[1]])
mask = mask_one

for i, cond in enumerate(stim_names.keys()):
    print("working on " + cond)
    with h5py.File(datafile, 'r') as h:
        all_subs = h[cond][()]
        [key for key in h.keys()]
        condition = h[group_key][()]
        group_1_indices = np.where(condition == group_1_definition)
        group_1 = np.take(all_subs, group_1_indices, axis=0)[0]
        group_2_indices = np.where(condition == group_2_definition)
        group_2 = np.take(all_subs, group_2_indices, axis=0)[0]

    prop_group_1 = np.nanmean(binarize(group_1.copy()), axis=0)
    masked_group_1 = np.ma.masked_where(mask != 1, prop_group_1)

    prop_group_2 = np.nanmean(binarize(group_2.copy()), axis=0)
    masked_group_2= np.ma.masked_where(mask != 1, prop_group_2)

    twosamp_t, twosamp_p = compare_groups(group_1, group_2)
    twosamp_p_corrected, twosamp_reject = p_adj_maps(twosamp_p, mask=mask, method='fdr_bh')
    twosamp_p_corrected[np.isnan(twosamp_p_corrected)] = 1

    twosamp_t[twosamp_p_corrected > 0.05] = 0
    masked_twosamp = twosamp_t.copy()
    masked_twosamp[mask != 1 ] = 0

    subplot_n_row_1 = i+1
    subplot_n_row_2 = 8+i
    subplot_n_row_3 = 15+i

    ax1 = plt.subplot(3,7,subplot_n_row_1)
    im1 = ax1.imshow(masked_group_1, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(stim_names[cond][0], size=30)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_axis_off()

    ax2 = plt.subplot(3,7,subplot_n_row_2)
    ax2.imshow(masked_group_2, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_axis_off()

    ax3 = plt.subplot(3,7,subplot_n_row_3)
    im3 = ax3.imshow(masked_twosamp, cmap=newcmp, vmin=-8, vmax=8)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_axis_off()


plt.tight_layout()
fig.subplots_adjust(wspace=-0.05, hspace=0, left=0.05, right=0.9)

[[x00,y00],[x01,y01]] = ax1.get_position().get_points()
[[x10,y10],[x11,y11]] = ax2.get_position().get_points()
pad = 0.015; width = 0.02
cbar_ax = fig.add_axes([x11+pad, y10+pad, width, y01-y10-2*pad])
axcb = fig.colorbar(im1, cax=cbar_ax)
axcb.set_label(label='Proportion of subjects', fontsize=20)
axcb.ax.tick_params(labelsize=20)

[[x20,y20],[x02,y02]] = ax3.get_position().get_points()
cbar2_ax = fig.add_axes([x02+pad, y20+pad, width, y02-y20-2*pad])
ax2cb = fig.colorbar(im3, cax=cbar2_ax)
ax2cb.set_label(label='Difference', fontsize=20)
ax2cb.ax.tick_params(labelsize=20)
ax2cb.ax.set_title(f"{group_1_name} > {group_2_name}", fontsize=20)
ax2cb.ax.set_xlabel(f"{group_2_name} > {group_1_name}", fontsize=20)

plt.gcf().text(0.03, 0.74, group_1_name, fontsize=24, rotation=90)
plt.gcf().text(0.03, 0.47, group_2_name, fontsize=24, rotation=90)
plt.gcf().text(0.03, 0.15, "Difference", fontsize=24, rotation=90)

plt.savefig(outfilename)
plt.close()

