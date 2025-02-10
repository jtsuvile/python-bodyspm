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
#
dataloc_lbp = '/Volumes/Shield1/kipupotilaat/data/stockholm/processed/lbp/'
dataloc_fibro = '/Volumes/Shield1/kipupotilaat/data/stockholm/processed/fibro/'
dataloc_controls = '/Volumes/Shield1/kipupotilaat/data/stockholm/controls/all/'
outfilename = '/Users/juusu53/Documents/projects/kipupotilaat/stockholm/figures/emotions_healthy_lbp_fibro_ALF.png'
suptitle = 'Average emotions'


datafile_fibro = get_latest_datafile(dataloc_fibro)
datafile_lbp = get_latest_datafile(dataloc_lbp)
datafile_controls = get_latest_datafile(dataloc_controls)

maskloc = '/Users/juusu53/Documents/projects/kipupotilaat/python_code/sample_data/'

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

all_figs = np.zeros([mask_one.shape[0], mask_one.shape[1]])
mask = mask_one

for i, cond in enumerate(stim_names.keys()):
    print("working on " + cond)
    with h5py.File(datafile_lbp, 'r') as h:
        lbp = h[cond][()]

    with h5py.File(datafile_fibro, 'r') as c:
        fibro = c[cond][()]

    with h5py.File(datafile_controls, 'r') as b:
        control = b[cond][()]

    

    prop_fibro = np.nanmean(binarize(fibro.copy()), axis=0)
    masked_fibro= np.ma.masked_where(mask != 1,prop_fibro)

    prop_lbp = np.nanmean(binarize(lbp.copy()), axis=0)
    masked_lbp = np.ma.masked_where(mask != 1, prop_lbp)

    prop_control = np.nanmean(binarize(control.copy()), axis=0)
    masked_control = np.ma.masked_where(mask != 1, prop_control)


    subplot_n_row_1 = i+1
    subplot_n_row_2 = 8+i
    subplot_n_row_3 = 15+i

    ax1 = plt.subplot(3,7,subplot_n_row_1)
    im1 = ax1.imshow(masked_control, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(stim_names[cond][0], size=30)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_axis_off()

    ax2 = plt.subplot(3,7,subplot_n_row_2)
    ax2.imshow(masked_fibro, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_axis_off()

    ax3 = plt.subplot(3,7,subplot_n_row_3)
    im3 = ax3.imshow(masked_lbp, cmap=cmap, vmin=vmin, vmax=vmax)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_axis_off()


plt.tight_layout()
fig.subplots_adjust(wspace=-0.05, hspace=0, left=0.05, right=0.9)

[[x00,y00],[x01,y01]] = ax1.get_position().get_points()
[[x10,y10],[x11,y11]] = ax3.get_position().get_points()
pad = 0.015; width = 0.02
cbar_ax = fig.add_axes([x11+pad, y10+pad, width, y01-y10-2*pad])
axcb = fig.colorbar(im1, cax=cbar_ax)
axcb.set_label(label='Proportion of subjects', fontsize=24)
axcb.ax.tick_params(labelsize=20)

plt.gcf().text(0.03, 0.40, "Fibromyalgia patients", fontsize=24, rotation=90)
plt.gcf().text(0.03, 0.12, "CLBP patients", fontsize=24, rotation=90)
plt.gcf().text(0.03, 0.74, "Pain-free controls", fontsize=24, rotation=90)

plt.savefig(outfilename)
plt.close()

