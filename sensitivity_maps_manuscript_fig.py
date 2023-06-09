from bodyfunctions import *
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

figloc = '/m/nbe/scratch/socbrain/kipupotilaat/figures/endometriosis/'
maskloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/'
dataloc = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/processed'
datafile = get_latest_datafile(dataloc)

dataloc_controls = '/m/nbe/scratch/socbrain/kipupotilaat/data/endometriosis/matched_controls'
datafile_controls = get_latest_datafile(dataloc_controls)

mask_fb = read_in_mask(maskloc + 'mask_front_new.png', maskloc + 'mask_back_new.png')

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

stim_names = {'sensitivity_0': ['Tactile sensitivity', 1],
              'sensitivity_1': ['Nociceptive sensitivity', 1],
              'sensitivity_2': ['Hedonic sensitivity', 1]}

mask = mask_fb
cmap = 'hot'
vmin = 0
vmax = 1
fig = plt.figure(figsize=(20, 25))

hotcool = cm.get_cmap('bwr', 256)
newcolors = hotcool(np.linspace(0, 1, 256))
outlinecolor = np.array([100/256, 100/256, 100/256, 1])
newcolors = np.vstack((outlinecolor, newcolors))
newcmp = ListedColormap(newcolors)

# Visualise group differences

for i, cond in enumerate(stim_names.keys()):
    print("working on " + cond)
    with h5py.File(datafile, 'r') as h:
        kipu = h[cond][()]

    with h5py.File(datafile_controls, 'r') as c:
        control = c[cond][()]

    prop_control = np.nanmean(binarize(control.copy()), axis=0)
    prop_control = prop_control + mask_array
    masked_control= np.ma.masked_where(mask != 1,prop_control)

    prop_kipu = np.nanmean(binarize(kipu.copy()), axis=0)
    prop_kipu = prop_kipu + mask_array
    masked_kipu = np.ma.masked_where(mask != 1, prop_kipu)

    twosamp_t, twosamp_p = compare_groups(kipu, control, testtype='z')
    twosamp_p_corrected, twosamp_reject = p_adj_maps(twosamp_p, mask=mask, method='fdr_bh')
    twosamp_p_corrected[np.isnan(twosamp_p_corrected)] = 1

    twosamp_t[twosamp_p_corrected > 0.05] = 0
    masked_twosamp = twosamp_t.copy()
    masked_twosamp[mask != 1 ] = 0
    masked_twosamp = masked_twosamp - mask_array*30

    if i==0:
        ax1 = plt.subplot(331)
        img1 = plt.imshow(masked_kipu, cmap=cmap, vmin=vmin, vmax=vmax)
        ax1.set_title('Tactile sensitivity', size=30)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_axis_off()

        ax3 = plt.subplot(334)
        img3 = plt.imshow(masked_control, cmap=cmap, vmin=vmin, vmax=vmax)
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_axis_off()

        ax5 = plt.subplot(337)
        img5 = plt.imshow(masked_twosamp, cmap=newcmp, vmin=-8, vmax=8)
        ax5.set_xticklabels([])
        ax5.set_yticklabels([])
        ax5.set_axis_off()

    elif i==1:
        ax2 = plt.subplot(332)
        img2 = plt.imshow(masked_kipu, cmap=cmap, vmin=vmin, vmax=vmax)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_title('Nociceptive sensitivity', size=30)
        ax2.set_axis_off()

        ax4 = plt.subplot(335)
        img4 = plt.imshow(masked_control, cmap=cmap, vmin=vmin, vmax=vmax)
        ax4.set_xticklabels([])
        ax4.set_yticklabels([])
        ax4.set_axis_off()

        ax6 = plt.subplot(338)
        img6 = plt.imshow(masked_twosamp, cmap=newcmp, vmin=-8, vmax=8)
        ax6.set_xticklabels([])
        ax6.set_yticklabels([])
        ax6.set_axis_off()

    elif i==2:
        ax2 = plt.subplot(333)
        img2 = plt.imshow(masked_kipu, cmap=cmap, vmin=vmin, vmax=vmax)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_title('Hedonic sensitivity', size=30)
        ax2.set_axis_off()

        ax4 = plt.subplot(336)
        img4 = plt.imshow(masked_control, cmap=cmap, vmin=vmin, vmax=vmax)
        ax4.set_xticklabels([])
        ax4.set_yticklabels([])
        ax4.set_axis_off()

        ax6 = plt.subplot(339)
        img6 = plt.imshow(masked_twosamp, cmap=newcmp, vmin=-8, vmax=8)
        ax6.set_xticklabels([])
        ax6.set_yticklabels([])
        ax6.set_axis_off()

plt.tight_layout()
fig.subplots_adjust(wspace=-0.05, hspace=0, left=0.05, right=0.9)

[[x00,y00],[x01,y01]] = ax2.get_position().get_points()
[[x10,y10],[x11,y11]] = ax4.get_position().get_points()
pad = 0.02; width = 0.025
cbar_ax = fig.add_axes([x11+pad, y10+pad, width, y01-y10-2*pad])
axcb = fig.colorbar(img1, cax=cbar_ax)
axcb.set_label(label='Proportion of subjects', fontsize=20)
axcb.ax.tick_params(labelsize=20)

[[x20,y20],[x02,y02]] = ax6.get_position().get_points()
cbar2_ax = fig.add_axes([x02+pad, y20+pad, width, y02-y20-2*pad])
ax2cb = fig.colorbar(img6, cax=cbar2_ax)
ax2cb.set_label(label='Difference', fontsize=20)
ax2cb.ax.tick_params(labelsize=20)
ax2cb.ax.set_title('pain >\ncontrol', fontsize=20)
ax2cb.ax.set_xlabel('control >\npain', fontsize=20)

plt.gcf().text(0.03, 0.85, "Patients", fontsize=24, rotation=90)
plt.gcf().text(0.03, 0.47, "Pain-free controls", fontsize=24, rotation=90)
plt.gcf().text(0.03, 0.18, "Difference", fontsize=24, rotation=90)

plt.savefig(figloc+'sensitivity_location_controls_pain_manuscript_fig.png')
plt.close()