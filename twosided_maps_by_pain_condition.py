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

dataloc_crps = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/crps/'
dataloc_lbp = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/lbp/'
dataloc_fibro = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/fibro/'
dataloc_np = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/np/'

datafile_crps = get_latest_datafile(dataloc_crps)
datafile_lbp = get_latest_datafile(dataloc_lbp)
datafile_fibro = get_latest_datafile(dataloc_fibro)
datafile_np = get_latest_datafile(dataloc_np)

cond = 'sensitivity_2'
outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/figures/compare_pain_locations_by_pain_type_' + cond + '.png'
suptitle = 'Hedonic sensitivity'


maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'

#stim_names = {'pain_0': ['current pain', 1], 'pain_1': ['chonic pain', 1]}
pain_names = ['CRPS','NP','LBP','fibromyalgia']
n_maps = 4
fig, axs = plt.subplots(1,n_maps+1, figsize=(11, 6), facecolor='w', edgecolor='k')

# outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/figures/all_pain_patients_sensitivity.png'
# suptitle = 'Sensitivity maps, pain patients'
# stim_names = {'sensitivity_0': ['tactile sensitivity',1],
#               'sensitivity_1': ['nociceptive sensitivity',1], 'sensitivity_2': ['hedonic sensitivity',1]}
# n_maps = len(stim_names.keys())
# fig, axs = plt.subplots(1,n_maps+1, figsize=(15, 6.5), facecolor='w', edgecolor='k')


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

all_figs = np.zeros([4, mask_fb.shape[0], mask_fb.shape[1]])
all_n = np.zeros(4)

# figure out which crps patients need to have their plots flipped to show all pains on the left
with h5py.File(datafile_crps, 'r') as h:
    pain_chronic = h['pain_1'].value
    subids = h['subid'].value

left_side = np.sum(np.sum(np.concatenate((pain_chronic[:,:, 85:169], pain_chronic[:,:, 172:256]),1),2),1)
right_side = np.sum(np.sum(np.concatenate((pain_chronic[:, 1:85], pain_chronic[:, 256:340]),1),2),1)

patient_pain_left = left_side - right_side > 0
patients_to_flip = subids[~patient_pain_left] # pick patients whose pain is typically on the right side & flip them over

pain_chronic_flipped = np.vstack((np.flip(pain_chronic[~patient_pain_left, :, :], axis=2), pain_chronic[patient_pain_left,:,:]))



print('reading in ' + cond)
with h5py.File(datafile_crps, 'r') as h:
    data_crps = h[cond].value
    # Flip the maps that need to be flipped
    # data_crps = np.vstack((np.flip(data_crps[~patient_pain_left, :, :], axis=2), data_crps[patient_pain_left,:,:]))
    data_flipped = np.concatenate((np.flip(data_crps[patient_pain_left, :, 0:171], axis=2),
                                      np.flip(data_crps[patient_pain_left, :, 170:341], axis=2)), axis=2)
    data_crps = np.vstack((data_flipped, data_crps[~patient_pain_left,:,:]))
all_figs[0, :, :] = np.nanmean(binarize(data_crps.copy()), axis=0)
all_n[0] = np.count_nonzero(~np.isnan(data_crps[:, 1, 1]))
with h5py.File(datafile_np, 'r') as h:
    data_np = h[cond].value
all_figs[1, :, :] = np.nanmean(binarize(data_np.copy()), axis=0)
all_n[1] = np.count_nonzero(~np.isnan(data_np[:, 1, 1]))
with h5py.File(datafile_lbp, 'r') as h:
    data_lbp = h[cond].value
all_figs[2, :, :] = np.nanmean(binarize(data_lbp.copy()), axis=0)
all_n[2] = np.count_nonzero(~np.isnan(data_lbp[:, 1, 1]))
with h5py.File(datafile_fibro, 'r') as h:
    data_fibro = h[cond].value
all_figs[3, :, :] = np.nanmean(binarize(data_fibro.copy()), axis=0)
all_n[3] = np.count_nonzero(~np.isnan(data_fibro[:, 1, 1]))

#
cmap = plt.cm.get_cmap('hot', 256)

vmin = 0
vmax = 0.8

axs = axs.ravel()

for i, cond in enumerate(pain_names):
    masked_data = np.ma.masked_where(mask_fb != 1,all_figs[i,:,:])
    masked_outlined = np.add(masked_data, outline_fb)
    im = axs[i].imshow(masked_outlined, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[i].set_axis_off()
    axs[i].set_title(cond + '\n n = ' + str(int(all_n[i])),fontsize=18)

divider = make_axes_locatable(axs[n_maps])
cax = divider.append_axes('left', size='10%', pad="2%")
axs[n_maps].set_axis_off()
fig.colorbar(im, cax=cax, orientation='vertical')
fig.suptitle(suptitle, size=20, va='top')
#plt.show()
plt.savefig(outfilename)
plt.close()




#
# rois_with_outline = sensitivity_all + np.hstack((outline_front_better, outline_back_better))

# data_crps = h[cond].value
# mean_left = np.nanmean(binarize(data_crps[patient_pain_left,:,:]), axis=0)
# mean_right = np.nanmean(binarize(data_crps[~patient_pain_left,:,:]), axis=0)
# data_made_right = np.concatenate((np.flip(data_crps[patient_pain_left, :, 0:171], axis=2), np.flip(data_crps[patient_pain_left, :, 170:341], axis=2)), axis=2)
# corrected_mean = np.nanmean(binarize(np.vstack((data_made_right, data_crps[~patient_pain_left,:,:]))), axis=0)
#
# fig, axs = plt.subplots(1, 3, figsize=(11, 8), facecolor='w', edgecolor='k')
# im = axs[0].imshow(mean_left, cmap=cmap, vmin=vmin, vmax=vmax)
# axs[0].set_axis_off()
# axs[0].set_title('avg left', fontsize=18)
# im = axs[1].imshow(mean_right, cmap=cmap, vmin=vmin, vmax=vmax)
# axs[1].set_axis_off()
# axs[1].set_title('avg right', fontsize=18)
# im = axs[2].imshow(corrected_mean, cmap=cmap, vmin=vmin, vmax=vmax)
# axs[2].set_axis_off()
# axs[2].set_title('flipped to right', fontsize=18)
#
#
# cax = divider.append_axes('left', size='10%', pad="2%")
# fig.colorbar(im, cax=cax, orientation='vertical')
#
# plt.savefig('/m/nbe/scratch/socbrain/kipupotilaat/figures/test_crps_flipping.png')
# plt.close()
