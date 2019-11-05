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


figloc = '/m/nbe/scratch/socbrain/kipupotilaat/figures/'
maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'
dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/helsinki/processed/'
datafile = get_latest_datafile(dataloc)

dataloc_controls = '/m/nbe/scratch/socbrain/kipupotilaat/data/controls/processed/matched_controls/'
datafile_controls = get_latest_datafile(dataloc_controls)

mask_fb = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')
mask_one = read_in_mask(maskloc + 'mask_front_new.png')

stim_names = {'emotions_0': ['sadness', 0], 'emotions_1': ['happiness', 0], 'emotions_2': ['anger', 0],
              'emotions_3': ['surprise', 0], 'emotions_4': ['fear', 0], 'emotions_5': ['disgust', 0],
              'emotions_6': ['neutral', 0],
              'pain_0': ['acute pain', 1], 'pain_1': ['chonic_pain', 1], 'sensitivity_0': ['tactile sensitivity', 1],
              'sensitivity_1': ['nociceptive sensitivity', 1], 'sensitivity_2': ['hedonic sensitivity', 1]}


hot = plt.cm.get_cmap('hot', 256)
new_cols = hot(np.linspace(0, 1, 256))
cold = np.hstack((np.fliplr(new_cols[:,0:3]),new_cols[:,3][:,None]))
cmap = ListedColormap(cold)

# Visualise group differences

for i, cond in enumerate(stim_names.keys()):
    print(cond)
    with h5py.File(datafile, 'r') as h:
        kipu = h[cond].value

    with h5py.File(datafile_controls, 'r') as c:
        control = c[cond].value

    if stim_names[cond][1] == 1:
        mask = mask_fb
        fig = plt.figure(figsize=(25, 10))
    else:
        mask = mask_one
        fig= plt.figure(figsize=(14,10))

    control_t = np.var(control,axis=0)
    masked_control= np.ma.masked_where(mask != 1,control_t)

    kipu_t = np.var(kipu,axis=0)#np.nanmean(binarize(pain), axis=0)
    masked_kipu= np.ma.masked_where(mask != 1,kipu_t)

    twosamp_t = np.zeros(kipu.shape[1:])
    twosamp_p = np.ones(kipu.shape[1:])

    # ugh, this is silly and slow. fix up later
    for x in range(0,twosamp_t.shape[0]-1):
        for y in range(0, twosamp_t.shape[1]-1):
            res = stats.levene(kipu[:,x,y], control[:,x,y], center='median')
            twosamp_t[x,y] = res[0]
            twosamp_p[x,y] = res[1]

    twosamp_p_corrected, twosamp_reject = p_adj_maps(twosamp_p, mask=mask, method='fdr_bh')
    #twosamp_p_corrected = twosamp_p
    twosamp_p_corrected[np.isnan(twosamp_p_corrected)] = 1
    twosamp_t_no_fdr = twosamp_t.copy()

    twosamp_t[twosamp_p_corrected > 0.05] = 0
    masked_twosamp = np.ma.masked_where(mask != 1, twosamp_t)

    twosamp_t_no_fdr[twosamp_p > 0.05] = 0
    masked_twosamp_no_fdr = np.ma.masked_where(mask != 1, twosamp_t_no_fdr)

    # change colormap for non-comparison to something more robust
    vmin = 0
    vmax1 = 0.0015
    vmax2 = 15

    ax1 = plt.subplot(142)
    img1 = plt.imshow(masked_kipu, cmap=cmap, vmin=vmin, vmax=vmax1)
    ax1.title.set_text('pain patients')
    fig.colorbar(img1,fraction=0.046, pad=0.04)
    ax1.axis('off')

    ax2 = plt.subplot(141)
    img2 = plt.imshow(masked_control, cmap=cmap, vmin=vmin, vmax=vmax1)
    ax2.title.set_text('controls')
    fig.colorbar(img2, fraction=0.046, pad=0.04)
    ax2.axis('off')

    ax3 = plt.subplot(143)
    img3 = plt.imshow(masked_twosamp, cmap='cividis', vmin=vmin, vmax=vmax2)
    ax3.title.set_text('Difference')
    fig.colorbar(img3, fraction=0.046, pad=0.04)
    ax3.axis('off')

    ax4 = plt.subplot(144)
    img4 = plt.imshow(masked_twosamp_no_fdr, cmap='cividis', vmin=vmin, vmax=vmax2)
    ax4.title.set_text('Difference, no FDR')
    fig.colorbar(img4, fraction=0.046, pad=0.04)
    ax4.axis('off')

    #
    fig.suptitle(stim_names[cond][0], size=20, va='top')
    #plt.show()
    plt.savefig(figloc+cond+'_controls_pain_variance_pixelwise.png')
    plt.close()

