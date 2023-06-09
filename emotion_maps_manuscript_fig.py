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
#
dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/matched_controls/'
dataloc1 = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/processed/'
outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/figures/endometriosis/emotions_endo_control.png'
suptitle = 'Average emotions'


datafile1 = get_latest_datafile(dataloc1)
datafile = get_latest_datafile(dataloc)

maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'

stim_names = {'emotions_2': ['Anger', 0],'emotions_4': ['Fear', 0],  'emotions_5': ['Disgust', 0],
              'emotions_1': ['Happiness', 0], 'emotions_0': ['Sadness', 0],
              'emotions_3': ['Surprise', 0], 'emotions_6': ['Neutral', 0]}

mask_fb = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')
mask_one = read_in_mask(maskloc + 'mask_front_new.png')

hot = plt.cm.get_cmap('hot', 256)
new_cols = hot(np.linspace(0, 1, 256))

cold = np.hstack((np.fliplr(new_cols[:,0:3]),new_cols[:,3][:,None]))
newcolors = np.vstack((np.flipud(cold), new_cols))
newcolors = np.delete(newcolors, np.arange(200, 312, 2), 0)
cmap = ListedColormap(newcolors)
#cmap = 'coolwarm'
vmin = -1
vmax = 1

fig, axs = plt.subplots(2,7, figsize=(25, 17), facecolor='w', edgecolor='k')
axs = axs.ravel()
all_figs = np.zeros([mask_one.shape[0], mask_one.shape[1]])

for i, cond in enumerate(stim_names.keys()):
    for j, file in enumerate([datafile1, datafile]):
        print('reading in ' + cond)
        with h5py.File(file, 'r') as h:
            data = h[cond][()]
            all_n = np.count_nonzero(~np.isnan(data[:,1,1]))
            all_figs = np.nanmean(binarize(data.copy()), axis=0)
            masked_data = np.ma.masked_where(mask_one != 1,all_figs)
        if j == 0:
            imind = i
        else:
            imind = i+7
        im = axs[imind].imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[imind].set_xticklabels([])
        axs[imind].set_yticklabels([])
        axs[imind].set_axis_off()
        if j==0:
            axs[imind].set_title(stim_names[cond][0],fontsize=30)

fig.subplots_adjust(wspace=0, hspace=0)
[[x00,y00],[x01,y01]] = axs[6].get_position().get_points()
[[x10,y10],[x11,y11]] = axs[13].get_position().get_points()
pad = 0.02; width = 0.01
cbar_ax = fig.add_axes([x11+pad, y10+pad, width, y01-y10-2*pad])
axcb = fig.colorbar(im, cax=cbar_ax)
axcb.set_label(label='Proportion of subjects', fontsize=20, labelpad=-120)
axcb.ax.tick_params(labelsize=20)


plt.gcf().text(0.11, 0.65, "Pain patients", fontsize=24, rotation=90)
plt.gcf().text(0.11, 0.25, "Matched controls", fontsize=24, rotation=90)
plt.gcf().text(0.88, 0.875, "Activation", fontsize=20)
plt.gcf().text(0.88, 0.1, "Inactivation", fontsize=20)

plt.savefig(outfilename)
plt.close()

