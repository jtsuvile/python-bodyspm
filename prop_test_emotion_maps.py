from classdefinitions import Subject, Stimuli
from bodyfunctions import *
import h5py
import numpy as np
import pandas as pd
from scipy import stats
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap



figloc = '/m/nbe/scratch/socbrain/kipupotilaat/figures/KI/'
maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'
dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/stockholm/processed/fibro/'
datafile = get_latest_datafile(dataloc)

dataloc_controls = '/m/nbe/scratch/socbrain/kipupotilaat/data/stockholm/processed/lbp/'
datafile_controls = get_latest_datafile(dataloc_controls)

mask_one = read_in_mask(maskloc + 'mask_front_new.png')

stim_names = {'emotions_0': ['sadness', 0], 'emotions_1': ['happiness', 0], 'emotions_2': ['anger', 0],
              'emotions_3': ['surprise', 0], 'emotions_4': ['fear', 0], 'emotions_5': ['disgust', 0],
              'emotions_6': ['neutral', 0]}

hot = plt.cm.get_cmap('hot', 256)
new_cols = hot(np.linspace(0, 1, 256))

cold = np.hstack((np.fliplr(new_cols[:,0:3]),new_cols[:,3][:,None]))
newcolors = np.vstack((np.flipud(cold), new_cols))
newcolors = np.delete(newcolors, np.arange(200, 312, 2), 0)


# Visualise group differences

for i, cond in enumerate(stim_names.keys()):

    with h5py.File(datafile, 'r') as h:
        kipu = h[cond].value

    with h5py.File(datafile_controls, 'r') as c:
        control = c[cond].value

    mask = mask_one
    cmap = ListedColormap(newcolors)
    vmin = -1
    vmax = 1
    fig= plt.figure(figsize=(20,10))

    control_avg = np.nanmean(binarize(control.copy()), axis=0)
    masked_control = np.ma.masked_where(mask != 1, control_avg)

    kipu_avg = np.nanmean(binarize(kipu.copy()), axis=0)
    masked_kipu = np.ma.masked_where(mask != 1, kipu_avg)

    #split into pos and neg maps
    kipu_pos = binarize(kipu.copy())
    kipu_pos[kipu_pos < - 0.5 ] = 0

    kipu_neg = binarize(kipu.copy())
    kipu_neg[kipu_neg > 0.5] = 0
    kipu_neg = kipu_neg*-1

    control_pos = binarize(control.copy())
    control_pos[control_pos < - 0.5 ] = 0

    control_neg = binarize(control.copy())
    control_neg[control_neg > 0.5] = 0
    control_neg = control_neg * -1

    print('Using z test of proportions')
    twosamp_t_pos, twosamp_p_pos = compare_groups(kipu_pos, control_pos, testtype='z')
    twosamp_t_neg, twosamp_p_neg = compare_groups(kipu_neg, control_neg, testtype='z')

    twosamp_p_corrected_pos, twosamp_reject_pos = p_adj_maps(twosamp_p_pos, mask=mask, method='fdr_bh')
    twosamp_p_corrected_neg, twosamp_reject_neg = p_adj_maps(twosamp_p_neg, mask=mask, method='fdr_bh')

    twosamp_p_corrected_pos[np.isnan(twosamp_p_corrected_pos)] = 1
    twosamp_t_no_fdr_pos = twosamp_t_pos.copy()

    twosamp_p_corrected_neg[np.isnan(twosamp_p_corrected_neg)] = 1
    twosamp_t_no_fdr_neg = twosamp_t_neg.copy()

    twosamp_t_pos[twosamp_p_corrected_pos > 0.05] = 0
    masked_twosamp_pos = np.ma.masked_where(mask != 1, twosamp_t_pos)

    twosamp_t_neg[twosamp_p_corrected_neg > 0.05] = 0
    masked_twosamp_neg = np.ma.masked_where(mask != 1, twosamp_t_neg)

    twosamp_t_no_fdr_pos[twosamp_p_pos > 0.05] = 0
    masked_twosamp_no_fdr_pos = np.ma.masked_where(mask != 1, twosamp_t_no_fdr_pos)

    twosamp_t_no_fdr_neg[twosamp_p_neg > 0.05] = 0
    masked_twosamp_no_fdr_neg = np.ma.masked_where(mask != 1, twosamp_t_no_fdr_neg)

    ax1 = plt.subplot(162)
    img1 = plt.imshow(masked_kipu, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.title.set_text('Fibromyalgia patients')
    fig.colorbar(img1,fraction=0.046, pad=0.04)
    ax1.axis('off')

    ax2 = plt.subplot(161)
    img2 = plt.imshow(masked_control, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.title.set_text('LBP patients')
    fig.colorbar(img2, fraction=0.046, pad=0.04)
    ax2.axis('off')

    ax3 = plt.subplot(163)
    img3 = plt.imshow(masked_twosamp_pos, cmap='bwr', vmin=-8, vmax=8)
    ax3.title.set_text('Difference pos')
    fig.colorbar(img3, fraction=0.046, pad=0.04)
    ax3.axis('off')

    ax4 = plt.subplot(164)
    img4 = plt.imshow(masked_twosamp_no_fdr_pos, cmap='bwr', vmin=-8, vmax=8)
    ax4.title.set_text('Difference pos, no FDR')
    fig.colorbar(img4, fraction=0.046, pad=0.04)
    ax4.axis('off')

    ax3 = plt.subplot(165)
    img3 = plt.imshow(masked_twosamp_neg, cmap='bwr', vmin=-8, vmax=8)
    ax3.title.set_text('Difference neg')
    fig.colorbar(img3, fraction=0.046, pad=0.04)
    ax3.axis('off')

    ax4 = plt.subplot(166)
    img4 = plt.imshow(masked_twosamp_no_fdr_neg, cmap='bwr', vmin=-8, vmax=8)
    ax4.title.set_text('Difference neg, no FDR')
    fig.colorbar(img4, fraction=0.046, pad=0.04)
    ax4.axis('off')

    #
    fig.suptitle(stim_names[cond][0], size=20, va='top')
    #plt.show()
    plt.savefig(figloc+cond+'_controls_pain_pixelwise_proportions_only.png')
    plt.close()
