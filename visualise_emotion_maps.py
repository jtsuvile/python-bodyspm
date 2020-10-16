from bodyfunctions import *
import h5py
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from operator import add
from mpl_toolkits.axes_grid1 import make_axes_locatable
#
# dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/matched_controls/'
# outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/figures/matched_controls_emotion_activations_new_order.png'
# suptitle = 'Average emotions, matched controls'

# dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/'
# outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/figures/all_controls_emotion_activations_new_order.png'
# suptitle = 'Average emotions, all controls'

# dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/stockholm/processed/'
# outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/figures/KI/lbp_emotion_activations.png'
# suptitle = 'Average emotions, lower back pain patients from KI'
#
# dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/stockholm/processed/'
# outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/figures/KI/fibro_emotion_activations.png'
# suptitle = 'Average emotions, fibromyalgia patients from KI'

dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/stockholm/processed/'
outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/figures/KI/karolinska_emotion_activations.png'
suptitle = 'Average emotions, all patients from KI'

datafile = get_latest_datafile(dataloc)
maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'

stim_names = {'emotions_2': ['anger', 0],'emotions_4': ['fear', 0],  'emotions_5': ['disgust', 0],
              'emotions_1': ['happiness', 0], 'emotions_0': ['sadness', 0],
              'emotions_3': ['surprise', 0], 'emotions_6': ['neutral', 0]}

# outline_front = scipy.misc.imread(maskloc + 'outline_front.png', flatten=True, mode='L')
# outline_back = scipy.misc.imread(maskloc + 'outline_back.png', flatten=True, mode='L')
#
# outline_back_better = outline_back.copy()
# outline_back_better[outline_back_better <= 20] = 0
# outline_back_better[outline_back_better > 20] = 1
# outline_front_better = outline_front.copy()
# outline_front_better[outline_front_better <= 20] = 0
# outline_front_better[outline_front_better > 20] = 1

mask_fb = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')
mask_one = read_in_mask(maskloc + 'mask_front_new.png')

all_figs = np.zeros([len(stim_names), mask_one.shape[0], mask_one.shape[1]])
all_n = np.zeros(len(stim_names))

for i, cond in enumerate(stim_names.keys()):
    print('reading in ' + cond)
    with h5py.File(datafile, 'r') as h:
        data = h[cond].value
        # kipu_diagnoses = list(h['groups'])
        #crps_indices = np.asarray([x == 'LOWER_BACK' for x in kipu_diagnoses])
        #crps_indices = np.asarray([x == 'FIBROMYALGI' for x in kipu_diagnoses])
        #data_special = data[crps_indices,:,:]
        data_special = data
        all_n[i] = np.count_nonzero(~np.isnan(data[:,1,1]))
    all_figs[i, :, :] = np.nanmean(binarize_posneg(data_special.copy()), axis=0)


hot = plt.cm.get_cmap('hot', 256)
new_cols = hot(np.linspace(0, 1, 256))

cold = np.hstack((np.fliplr(new_cols[:,0:3]),new_cols[:,3][:,None]))
newcolors = np.vstack((np.flipud(cold), new_cols))
newcolors = np.delete(newcolors, np.arange(200, 312, 2), 0)
cmap = ListedColormap(newcolors)
#cmap = 'coolwarm'
vmin = -1
vmax = 1

fig, axs = plt.subplots(1,8, figsize=(25, 10), facecolor='w', edgecolor='k')
axs = axs.ravel()

for i, cond in enumerate(stim_names.keys()):
    masked_data = np.ma.masked_where(mask_one != 1,all_figs[i,:,:])
    im = axs[i].imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[i].set_axis_off()
    axs[i].set_title(stim_names[cond][0] + '\n n = ' + str(int(all_n[i])),fontsize=18)

divider = make_axes_locatable(axs[7])
cax = divider.append_axes('left', size='10%', pad="2%")
axs[7].set_axis_off()
fig.colorbar(im, cax=cax, orientation='vertical')
# fig.colorbar(img1,fraction=0.046, pad=0.04)
fig.suptitle(suptitle + '\n n = ' + str(data_special.shape[0]), size=20, va='top')
#fig.suptitle('n = ' + str(np.count_nonzero(~np.isnan(data[:,1,1]))))
#plt.show()
plt.savefig(outfilename)
plt.close()


# rois_with_outline = sensitivity_all + np.hstack((outline_front_better, outline_back_better))