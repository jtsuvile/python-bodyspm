from bodyfunctions import *
import h5py
import numpy as np
from PIL import Image
# from scipy import misc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from operator import add
from mpl_toolkits.axes_grid1 import make_axes_locatable
#
# dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/matched_controls/'
# outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/figures/matched_controls_pain_locations.png'
# suptitle = 'Pain locations, matched controls'
# outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/figures/matched_controls_sensitivity.png'
# suptitle = 'Sensitivity maps, matched controls'

# dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/'
# outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/figures/all_controls_pain_locations.png'
# suptitle = 'Pain locations, general population'
# outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/figures/all_controls_sensitivity.png'
# suptitle = 'Sensitivity maps, general population'

dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/'
outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/figures/misc_patients_pain_locations.png'
suptitle = 'Pain locations, non-CRPS patients'
# outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/figures/misc_pain_patients_sensitivity.png'
# suptitle = 'Sensitivity maps, non-CRPS patients'

datafile = get_latest_datafile(dataloc)
maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'

stim_names = {'pain_0': ['current pain', 1], 'pain_1': ['chonic pain', 1]}
# stim_names =  {'sensitivity_0': ['tactile sensitivity',1],
#  'sensitivity_1': ['nociceptive sensitivity',1], 'sensitivity_2': ['hedonic sensitivity',1]}

outline_front = np.asarray(Image.open(maskloc + 'outline_front.png'))[:,:,0]
outline_back = np.asarray(Image.open(maskloc + 'outline_back.png'))[:,:,0]

outline_back_better = outline_back.copy()
outline_back_better[outline_back_better <= 20] = 0
outline_back_better[outline_back_better > 20] = 1
outline_front_better = outline_front.copy()
outline_front_better[outline_front_better <= 20] = 0
outline_front_better[outline_front_better > 20] = 1
outline_fb = np.concatenate((outline_front_better, outline_back_better), axis=1)

mask_fb = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')

all_figs = np.zeros([len(stim_names), mask_fb.shape[0], mask_fb.shape[1]])

for i, cond in enumerate(stim_names.keys()):
    print('reading in ' + cond)
    with h5py.File(datafile, 'r') as h:
        data = h[cond].value
        kipu_diagnoses = list(h['groups'])
        crps_indices = np.asarray([x != 'CRPS' for x in kipu_diagnoses])
        data_special = data[crps_indices,:,:]
    all_figs[i, :, :] = np.nanmean(binarize_posneg(data_special.copy()), axis=0)


cmap = plt.cm.get_cmap('hot', 256)
n_maps = len(stim_names.keys())

vmin = 0
vmax = 0.8

fig, axs = plt.subplots(1,n_maps+1, figsize=(15, 6.5), facecolor='w', edgecolor='k')
axs = axs.ravel()

for i, cond in enumerate(stim_names.keys()):
    masked_data = np.ma.masked_where(mask_fb != 1,all_figs[i,:,:])
    masked_outlined = np.add(masked_data, outline_fb)
    im = axs[i].imshow(masked_outlined, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[i].set_axis_off()
    axs[i].set_title(stim_names[cond][0],fontsize=18)

divider = make_axes_locatable(axs[n_maps])
cax = divider.append_axes('left', size='10%', pad="2%")
axs[n_maps].set_axis_off()
fig.colorbar(im, cax=cax, orientation='vertical')
# fig.colorbar(img1,fraction=0.046, pad=0.04)
fig.suptitle(suptitle, size=20, va='top')
#plt.show()
plt.savefig(outfilename)
plt.close()


# rois_with_outline = sensitivity_all + np.hstack((outline_front_better, outline_back_better))