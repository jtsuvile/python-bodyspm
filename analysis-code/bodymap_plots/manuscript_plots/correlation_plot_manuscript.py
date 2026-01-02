import sys
import os
from skimage import io

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(root_dir)

import pandas as pd
import numpy as np
from bodyfunctions import binarize, correlate_maps, p_adj_maps
from bodyfunctions import get_latest_datafile, read_in_mask, align_data
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap


dataloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/processed/all'
tableloc = '/Volumes/Shield1/kipupotilaat/data/stockholm/intermediate/' +\
    'combined_data_patients.csv'
outdataloc = '/Users/juusu53/Documents/projects/kipupotilaat/stockholm/r_code/figures/'
map_family = 'emotions'

interesting_variables = ["pain_duration", "score_bdi", "score_stai_state", "score_stai_trait", 
                         "styrka_nu", "intensity_range", "Chronic pain_total_color"]

threshold = 0.001  # 0.001 for patients, 0.007 for controls

maskloc = '/Users/juusu53/Documents/projects/kipupotilaat/' +\
    'python_code/sample_data/'
datafile = get_latest_datafile(dataloc)

mask_one = read_in_mask(maskloc + 'mask_front_new.png')
mask_fb = read_in_mask(maskloc + 'mask_front_new.png',
                       maskloc + 'mask_back_new.png')

stim_names = {'emotions_0': ['Sadness', 0],
              'emotions_1': ['Happiness', 0],
              'emotions_2': ['Anger', 0],
              'emotions_3': ['Surprise', 0],
              'emotions_4': ['Fear', 0],
              'emotions_5': ['Disgust', 0],
              'emotions_6': ['Neutral', 0],
              'pain_0': ['Current', 1],
              'pain_1': ['Chonic', 1],
              'sensitivity_0': ['Tactile', 1],
              'sensitivity_1': ['Nociceptive', 1],
              'sensitivity_2': ['Hedonic', 1]}


if map_family == 'emotions':
    which_maps = ['emotions_2', 'emotions_4', 'emotions_5',
                  'emotions_1', 'emotions_0',
                  'emotions_3', 'emotions_6']
    analysis = 'spearmanr'
    figsize_x = 24
    figsize_y = 10
    colorbar_aspect = 30
    mask = mask_one
    cbar_font_size = 16
    title_font_size = 20
elif map_family == 'sensitivity':
    which_maps = ['sensitivity_0', 'sensitivity_1', 'sensitivity_2']
    analysis = 'pointbiserialr'
    figsize_x = 16
    figsize_y = 10
    colorbar_aspect = 15
    mask = mask_fb
    cbar_font_size = 16
    title_font_size = 14
elif map_family == 'pain':
    which_maps = ['pain_0', 'pain_1']
    analysis = 'pointbiserialr'
    figsize_x = 10
    figsize_y = 7
    mask = mask_fb
    colorbar_aspect = 15
    cbar_font_size = 12
    title_font_size = 14
else:
    sys.exit(f"map family {map_family} not recognized")

# read in outline
if mask.shape == mask_fb.shape:
    mask_array = io.imread(maskloc + 'kipu_traced_outline_front.png', as_gray=True)
    mask_array[mask_array < 1] = 0
    dims = mask_array.shape
    if len(dims) == 3:
        mask_array = mask_array[:, :, 0]
    mask_other_side = io.imread(maskloc + 'kipu_traced_outline_back.png', as_gray=True)
    mask_other_side[mask_other_side < 1] = 0
    dims = mask_other_side.shape
    if len(dims) == 3:
        mask_other_side = mask_other_side[:, :, 0]
    mask_array = np.concatenate((mask_array, mask_other_side), axis=1)
else:
    mask_array = io.imread(maskloc + 'kipu_traced_outline.png', as_gray=True)
    mask_array[mask_array < 1] = 0
    dims = mask_array.shape
    if len(dims) == 3:
        mask_array = mask_array[:, :, 0]


# define colormap
coolwarm = plt.cm.get_cmap('coolwarm', 256)
newcolors = coolwarm(np.linspace(0, 1, 256))
outlinecolor = np.array([100/256, 100/256, 100/256, 1])
newcolors = np.vstack((outlinecolor, newcolors))
cmap = ListedColormap(newcolors)

#cmap = 'coolwarm'

vmin = -1
vmax = 1

# read in data
df = pd.read_csv(tableloc)

for variablename in interesting_variables:
    fig = plt.figure(figsize=(figsize_x, figsize_y))
    gs = gridspec.GridSpec(1, len(which_maps) + 1, width_ratios=[1] * len(which_maps) + [0.05])
    for m, which_map in enumerate(which_maps):
        print(f"{which_map} x {variablename}")
        with h5py.File(datafile, 'r') as h:
            subs = h['subid'][()]
            fig_data = h[which_map][()]
        bodymap, aligned_df = align_data(subs, fig_data, df, 'internetnummer')
        corr_variable = list(aligned_df[variablename])
        bodymap_bin = binarize(bodymap.copy(), threshold=threshold)

        # run analysis
        result_map_r, result_map_p = correlate_maps(bodymap_bin,
                                                    corr_variable,
                                                    method=analysis)

        # make plots

        result_map_r_with_fdr = result_map_r.copy()

        result_map_p_corrected, twosamp_reject = \
            p_adj_maps(result_map_p, mask=mask, method='fdr_bh')

        result_map_r_with_fdr[result_map_p_corrected > 0.05] = 0
        result_map_r_with_fdr[np.isnan(result_map_p_corrected)] = 0

        masked_result_map_with_fdr = \
            np.ma.masked_where(mask != 1, result_map_r_with_fdr)
        masked_result_map_with_fdr_and_outline = masked_result_map_with_fdr - mask_array*30

        ax1 = plt.subplot(gs[m])
        img1 = plt.imshow(masked_result_map_with_fdr_and_outline, cmap=cmap, 
                          vmin=-1, vmax=1,
                          interpolation="nearest")
        ax1.set_title(f'{stim_names[which_map][0]}', fontsize=title_font_size)
        ax1.axis('off')

    # colorbar
    cbar_ax = plt.subplot(gs[-1])
    cbar = fig.colorbar(img1, cax=cbar_ax)
    cbar_ax.set_aspect(colorbar_aspect) 
    cbar.set_label('Correlation coefficient', fontsize=cbar_font_size)
    cbar_ax.yaxis.set_label_position('left')
    cbar_ax.tick_params(labelsize=cbar_font_size)
    plt.tight_layout()

    #fig.suptitle(f"{map_family} vs {variablename}")
    plt.savefig(outdataloc + '/' + map_family + '_' + variablename + '.png')
    plt.close()

