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

stim_names = {'sensitivity_0': ['Tactile sensitivity', 1],
              'sensitivity_1': ['Nociceptive sensitivity', 1],
              'sensitivity_2': ['Hedonic sensitivity', 1]}
outfilename = '/m/nbe/scratch/socbrain/kipupotilaat/figures/compare_pain_locations_by_pain_type_sensitivities.png'

maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'
pain_names = ['CRPS','NP','LBP','fibromyalgia']

# get supporting files
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

# define plot stuff
suptitle = ' '
n_maps = len(stim_names) * len(pain_names)
fig, axs = plt.subplots(len(stim_names),len(pain_names), figsize=(11, 11), facecolor='w', edgecolor='k')
cmap = plt.cm.get_cmap('hot', 256)
vmin = 0
vmax = 0.8
axs = axs.ravel()

# figure out which crps patients need to have their plots flipped to show all pains on the left
with h5py.File(datafile_crps, 'r') as h:
    pain_chronic = h['pain_1'].value
    subids = h['subid'].value

left_side = np.sum(np.sum(np.concatenate((pain_chronic[:,:, 85:169], pain_chronic[:,:, 172:256]),1),2),1)
right_side = np.sum(np.sum(np.concatenate((pain_chronic[:, 1:85], pain_chronic[:, 256:340]),1),2),1)

patient_pain_left = left_side - right_side > 0
patients_to_flip = subids[~patient_pain_left] # pick patients whose pain is typically on the right side & flip them over

pain_chronic_flipped = np.vstack((np.flip(pain_chronic[~patient_pain_left, :, :], axis=2), pain_chronic[patient_pain_left,:,:]))

all_figs = np.zeros([n_maps, mask_fb.shape[0], mask_fb.shape[1]])
all_n = np.zeros(n_maps)
all_titles = []
#collect data
location_counter = 0
for i, cond in enumerate(stim_names.keys()):
    print('reading in ' + cond)
    with h5py.File(datafile_crps, 'r') as h:
        data_crps = h[cond].value
        # Flip the maps that need to be flipped
        # data_crps = np.vstack((np.flip(data_crps[~patient_pain_left, :, :], axis=2), data_crps[patient_pain_left,:,:]))
        data_flipped = np.concatenate((np.flip(data_crps[patient_pain_left, :, 0:171], axis=2),
                                        np.flip(data_crps[patient_pain_left, :, 170:341], axis=2)), axis=2)
        data_crps = np.vstack((data_flipped, data_crps[~patient_pain_left,:,:]))
    all_figs[location_counter, :, :] = np.nanmean(binarize(data_crps.copy()), axis=0)
    all_n[location_counter] = np.count_nonzero(~np.isnan(data_crps[:, 1, 1]))
    if(i==0):
        title_text = 'CRPS \n n = ' + str(int(all_n[location_counter]))
    else:
        title_text = ''
    all_titles.append(title_text)
    location_counter += 1
    with h5py.File(datafile_np, 'r') as h:
        data_np = h[cond].value
    all_figs[location_counter, :, :] = np.nanmean(binarize(data_np.copy()), axis=0)
    all_n[location_counter] = np.count_nonzero(~np.isnan(data_np[:, 1, 1]))
    if(i==0):
        title_text = 'Neuropathic \n n = ' + str(int(all_n[location_counter]))
    else:
        title_text = ''
    all_titles.append(title_text)
    location_counter += 1
    with h5py.File(datafile_lbp, 'r') as h:
        data_lbp = h[cond].value
    all_figs[location_counter, :, :] = np.nanmean(binarize(data_lbp.copy()), axis=0)
    all_n[location_counter] = np.count_nonzero(~np.isnan(data_lbp[:, 1, 1]))
    if(i==0):
        title_text = 'LBP \n n = ' + str(int(all_n[location_counter]))
    else:
        title_text = ''
    all_titles.append(title_text)
    location_counter += 1
    with h5py.File(datafile_fibro, 'r') as h:
        data_fibro = h[cond].value
    all_figs[location_counter, :, :] = np.nanmean(binarize(data_fibro.copy()), axis=0)
    all_n[location_counter] = np.count_nonzero(~np.isnan(data_fibro[:, 1, 1]))
    if(i==0):
        title_text = 'Fibromyalgia \n n = ' + str(int(all_n[location_counter]))
    else:
        title_text = ''
    all_titles.append(title_text)
    location_counter += 1

#
for i in range(0,len(all_n)):
    map_loc = i
    masked_data = np.ma.masked_where(mask_fb != 1,all_figs[i,:,:])
    masked_outlined = np.add(masked_data, outline_fb)
    im = axs[map_loc].imshow(masked_outlined, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[map_loc].set_axis_off()
    axs[map_loc].set_title(all_titles[i],fontsize=18)


plt.tight_layout()
fig.subplots_adjust(wspace=-0.1, hspace=0, left=0.05, right=0.9)

[[x00,y00],[x01,y01]] = axs[3].get_position().get_points()
[[x10,y10],[x11,y11]] = axs[11].get_position().get_points()
pad = 0.02; width = 0.019
cbar_ax = fig.add_axes([x11+pad, y10+pad, width, y01-y10-2*pad])
axcb = fig.colorbar(im, cax=cbar_ax)
axcb.set_label(label='Proportion of subjects', fontsize=20)
axcb.ax.tick_params(labelsize=20)


plt.gcf().text(0.03, 0.83, 'Tactile', fontsize=24, rotation=90)
plt.gcf().text(0.03, 0.57, 'Nociceptive', fontsize=24, rotation=90)
plt.gcf().text(0.03, 0.25, "Hedonic", fontsize=24, rotation=90)
#plt.show()
plt.savefig(outfilename)
plt.close()


