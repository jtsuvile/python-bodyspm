from bodyfunctions import *
import numpy as np
import matplotlib.pyplot as plt

dataloc = '/Users/jtsuvile/Documents/projects/kipupotilaat/python_code_testing/'
data= np.genfromtxt('/Users/jtsuvile/Documents/projects/kipupotilaat/python_code_testing/test_sub_4/sensitivity_1_as_matrix.csv', delimiter=',')
mask_use = read_in_mask(dataloc + 'mask_front_new.png',dataloc + 'mask_back_new.png')

mask_indices = np.nonzero(mask_use)
inside_mask = data[mask_indices[0], mask_indices[1]]
this_map_copy = np.copy(data)
#this_map_copy[mask_indices[0],mask_indices[1]] = inside_mask+0.1

fig = plt.figure()
img = plt.imshow(this_map_copy)
fig.colorbar(img)
plt.show()

# a = np.arange(25)
# a_sq = np.reshape(a, (5,5))
# b = np.zeros((5,5))
# b[1:4,1:3] =1
#
# masked_vals = a_sq[b.astype(int)]
# masked_vals = masked_vals + 100
#
# a_sq[b==1] = masked_vals