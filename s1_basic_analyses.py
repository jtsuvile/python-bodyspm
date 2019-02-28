from classdefinitions import Subject, Stimuli
from bodyfunctions import combine_data
import pickle
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Read pickled dataset from file

dataloc = '/Users/jtsuvile/Documents/projects/kipupotilaat/python_code_testing/'
datafile = dataloc + 'full_dataset.pickle'
all_data = pickle.load(open(datafile, "rb" ))

alpha = 0.2 # for testing with the tiny data set, in real use you would set this to 0.05/0.01 or whatever alpha level you choose

# one sample t-test
statistics, pval = stats.ttest_1samp(all_data['emotions_1'], 0, nan_policy = 'omit', axis=0)

lim = max(np.nanmax(statistics), abs(np.nanmin(statistics)))

stats_to_show = statistics.copy()
pval[np.isnan(pval)] = 1
stats_to_show[pval>alpha] = 0

fig = plt.figure()
img = plt.imshow(stats_to_show, cmap='RdBu_r', vmin=-10, vmax=10)
fig.colorbar(img)
plt.show()