from bodyfunctions import *
import numpy as np
import matplotlib.pyplot as plt

dataloc = '/Users/juusu53/Documents/projects/kipupotilaat/python_code_testing/'
# data= np.genfromtxt('/Users/juusu53/Documents/projects/kipupotilaat/python_code_testing/test_sub_4/sensitivity_1_as_matrix.csv', delimiter=',')
# mask_use = read_in_mask(dataloc + 'mask_front_new.png',dataloc + 'mask_back_new.png')
#
# mask_indices = np.nonzero(mask_use)
# inside_mask = data[mask_indices[0], mask_indices[1]]
# this_map_copy = np.copy(data)
# #this_map_copy[mask_indices[0],mask_indices[1]] = inside_mask+0.1
#
# fig = plt.figure()
# img = plt.imshow(this_map_copy)
# fig.colorbar(img)
# plt.show()

# a = np.arange(25)
# a_sq = np.reshape(a, (5,5))
# b = np.zeros((5,5))
# b[1:4,1:3] =1
#
# masked_vals = a_sq[b.astype(int)]
# masked_vals = masked_vals + 100
#
# a_sq[b==1] = masked_vals

# TEST ROI FILE

data = np.genfromtxt('/Users/juusu53/Documents/projects/kipupotilaat/python_code_testing/test_sub_1/emotions_4_as_matrix.csv', delimiter=',')
rois = io.imread('/Users/juusu53/Documents/projects/kipupotilaat/kipu_ROI_new.png', as_gray=True, pilmode='L')
mask_use = read_in_mask(dataloc + 'mask_front_new.png')
color_defs = {'head': 26, 'shoulders':128, 'arms': 102, 'upper_torso': 51, 'lower_torso': 77, 'legs': 153, 'hands': 204, 'feet': 230}

unique_elements, counts_elements = np.unique(rois, return_counts=True)


data_show = data.copy() #rois.copy()
data_show[rois != color_defs['legs']] = 0

fig = plt.figure()
img = plt.imshow(data_show)
fig.colorbar(img)
plt.show()