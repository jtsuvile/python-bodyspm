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



figloc = '/Users/juusu53/Documents/projects/endometrioosi/figures/'
maskloc = '/Users/juusu53/Documents/projects/kipupotilaat/python_code/sample_data/'
dataloc = '/Volumes/Shield1/kipupotilaat/data/endometriosis/processed_peritoneal/'
datafile = get_latest_datafile(dataloc)

dataloc_controls = '/Volumes/Shield1/kipupotilaat/data/endometriosis/processed_other_mixed/'
datafile_controls = get_latest_datafile(dataloc_controls)

mask_fb = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')
mask_one = read_in_mask(maskloc + 'mask_front_new.png')

stim_names = {
    #'emotions_0': ['sadness', 0], 'emotions_1': ['happiness', 0], 'emotions_2': ['anger', 0],
    #          'emotions_3': ['surprise', 0], 'emotions_4': ['fear', 0], 'emotions_5': ['disgust', 0],
    #          'emotions_6': ['neutral', 0],
              'pain_0': ['current pain', 1], 'pain_1': ['chonic pain', 1], 
              #'sensitivity_0': ['tactile sensitivity', 1],
              #'sensitivity_1': ['nociceptive sensitivity', 1], 'sensitivity_2': ['hedonic sensitivity', 1]
              }

#stim_names = {'pain_0': ['current pain', 1], 'pain_1': ['chonic pain', 1]}

hot = plt.cm.get_cmap('hot', 256)
new_cols = hot(np.linspace(0, 1, 256))

cold = np.hstack((np.fliplr(new_cols[:,0:3]),new_cols[:,3][:,None]))
newcolors = np.vstack((np.flipud(cold), new_cols))
newcolors = np.delete(newcolors, np.arange(200, 312, 2), 0)



for i, cond in enumerate(stim_names.keys()):

    with h5py.File(datafile, 'r') as h:
        kipu = h[cond][()]

    with h5py.File(datafile_controls, 'r') as c:
        control = c[cond][()]

    if stim_names[cond][1] == 1:
        mask = mask_fb
        cmap = 'hot'
        vmin = 0
        vmax = 1
        fig = plt.figure(figsize=(25, 10))
    else:
        mask = mask_one
        #cmap = 'coolwarm'
        cmap = ListedColormap(newcolors)
        vmin = -1
        vmax = 1
        fig= plt.figure(figsize=(14,10))

    control_t = np.nanmean(binarize(control.copy()), axis=0)
    masked_control= np.ma.masked_where(mask != 1, control_t)

    kipu_t = np.nanmean(binarize(kipu.copy()), axis=0)
    masked_kipu= np.ma.masked_where(mask != 1, kipu_t)

    if stim_names[cond][1]==1:
        # something wrong with this code
        # print('Using z test of proportions')
        twosamp_t, twosamp_p = compare_groups(kipu, control, testtype='z')
    else:
        twosamp_t, twosamp_p = stats.ttest_ind(kipu, control, axis=0, nan_policy='omit')
        
    twosamp_p_corrected, twosamp_reject = p_adj_maps(twosamp_p, mask=mask, method='fdr_bh')
    #twosamp_p_corrected = twosamp_p
    # sns.displot(data=pd.DataFrame({"data": twosamp_p.ravel(),
    #                           "column": np.repeat(np.arange(twosamp_p.shape[0]), twosamp_p.shape[1])}),
    #        x="data", kde=True, color='blueviolet', height=3)
    # plt.savefig(figloc+cond+'_twosamp_p.png')
    # plt.close()
    ### Check np.isnan(twosamp_p_corrected) -> do we have too much stuff there?
    
    twosamp_p_corrected[np.isnan(twosamp_p_corrected)] = 1
    twosamp_t_no_fdr = twosamp_t.copy()

    twosamp_t[twosamp_p_corrected > 0.05] = 0
    masked_twosamp = np.ma.masked_where(mask != 1, twosamp_t)

    twosamp_t_no_fdr[twosamp_p > 0.05] = 0
    masked_twosamp_no_fdr = np.ma.masked_where(mask != 1, twosamp_t_no_fdr)

    ax1 = plt.subplot(142)
    img1 = plt.imshow(masked_kipu, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.title.set_text('Peritoneal')
    fig.colorbar(img1,fraction=0.046, pad=0.04)
    ax1.axis('off')

    ax2 = plt.subplot(141)
    img2 = plt.imshow(masked_control, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.title.set_text('Other/mixed')
    fig.colorbar(img2, fraction=0.046, pad=0.04)
    ax2.axis('off')

    ax3 = plt.subplot(143)
    img3 = plt.imshow(masked_twosamp, cmap='bwr', vmin=-8, vmax=8)
    ax3.title.set_text('Difference')
    fig.colorbar(img3, fraction=0.046, pad=0.04)
    ax3.axis('off')

    ax4 = plt.subplot(144)
    img4 = plt.imshow(masked_twosamp_no_fdr, cmap='bwr', vmin=-8, vmax=8)
    ax4.title.set_text('Difference, no FDR')
    fig.colorbar(img4, fraction=0.046, pad=0.04)
    ax4.axis('off')

    #
    fig.suptitle(stim_names[cond][0], size=20, va='top')
    #plt.show()
    plt.savefig(figloc+cond+'_peritoneal_other_pixelwise.png')
    plt.close()
