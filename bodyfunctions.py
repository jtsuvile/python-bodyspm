# imports
import numpy as np
import pandas as pd
from scipy import stats
from skimage import io
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
from classdefinitions import Subject, Stimuli
import pickle


def preprocess_subjects(subnums, indataloc, outdataloc, stimuli, bgfiles=None,fieldnames=None):
    """Reads in data from web interface output and writes the subjects out to .csv files (colouring data) and
    .json (other sub data). Also draws single subject's data into file for quality control.
    :param subnums : list of subject numbers to process
    :param indataloc : where the data from the online interface has been saved
    :param outdataloc : where you'd like for the data to be stored
    :param stimuli : an object of class Stimuli, with information about the stimulus file names and
    whether data has been collected as one- or twosided
    :param bgfiles: list, optional. If you want to read in data from a comma separated text file as background information,
    supply a list of files to include
    :param fieldnames: list of lists, optional. List for each background information file giving the names of the
    fields (used as keys in background info dictionary)

    Does not return anything, data are stored as files to outdataloc
    """

    for i, subnum in enumerate(subnums):
        print("preprocessing subject " +  str(subnum) + " which is " + str(i+1) + "/" + str(len(subnums)))
        # make subject
        sub = Subject(subnum)
        sub.read_data(indataloc, stimuli)
        # if files with background information have been defined, save values to sub.bginfo
        if bgfiles or fieldnames is not None:
            for j,file in enumerate(bgfiles):
                sub.read_bg(indataloc, bgfiles[j], fieldnames[j])
        # write to file
        sub.write_sub_to_file(outdataloc)
        # plot single subject data
        # sub.draw_sub_data(stimuli, outdataloc)
    print("writing out stimulus definitions to file")
    stimuli.write_stim_to_file(outdataloc)
    print('done with preprocessing the subjects')
    return


def binarize(data):
    """
    Change data from colouring (with blur) to binary 1/0 format. This is used in several analyses where data have
    to be in binary format, such as two sample z test

    NB: 0.007 chosen as limit based on what limit replicates coloring best in Aalto system (March 2019).
    The best value for this parameter will depend on brush & blur settings.
    If changed, I highly recommend visual inspection of the result against known colouring

    :param data: matrix with colouring data
    :return: same matrix, with coloured areas changed to 1 and non-coloured changed to 0
    """

    data[data > 0.007] = 1
    data[data <= 0.007] = 0
    return data


def combine_data(dataloc, subnums, save=False):
    """
    Combines a data set from subjects who have been written to file.

    :param dataloc: where the subject data files have been saved. Assumes .json files for subjects and
    stimuli are located in this folder
    :param subnums: which subjects to combine (list). Assumes one .json file per subject
    :param save: do you want the combined data set saved into file (pickled)? (Boolean)
    :return: combined data for the defined subjects.
    The data are stored in a dictionary, where each body map will be presented as N * X * Y numpy array,
    where N is length of subnums, and X and Y are the dimensions of the maps. The dictionary will have keys for
    each stimulus (with the value being 3-D Numpy array) and 'subids' (list of the subids in the same order as they appear
    in the data arrays).
    """

    stim = Stimuli(fileloc=dataloc, from_file=True)
    size_onesided = (522, 171)
    size_twosided = (522, 342)
    all_res = {}
    all_res['subids'] = subnums
    all_res['stimuli'] = stim
    all_res['bg'] = pd.DataFrame(index=subnums)
    # init empty arrays for data
    for key in stim.all.keys():
        if stim.all[key]['onesided']:
            all_res[key] = np.zeros((len(subnums), size_onesided[0], size_onesided[1]))
        else:
            all_res[key] = np.zeros((len(subnums), size_twosided[0], size_twosided[1]))
    # populate with data
    for j, subnum in enumerate(subnums):
        temp_sub = Subject(subnum)
        temp_sub.read_sub_from_file(dataloc)
        for key, value in temp_sub.data.items():
            all_res[key][j] = temp_sub.data[key]
        for bgkey, bgvalue in temp_sub.bginfo.items():
            if not bgkey in all_res['bg'].columns:
                all_res['bg'][bgkey] = np.nan
            all_res['bg'].loc[subnum, bgkey] = bgvalue
    print("combined all data successfully ")
    if save:
        filename = dataloc + '/full_dataset.pickle'
        with open(filename, 'wb') as handle:
            pickle.dump(all_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("saved pickle to " +dataloc + "/full_dataset.pickle")
    return all_res


def one_sample_t_test(data):
    """
    one sample t-test to see if coloured data is significantly more than 0

    :param data: a 3-D matrix of subject-wise colouring maps
    :return: statistics: t statistic for each pixel
    :return: p-value for each pixel (uncorrected)
    """
    statistics, pval = stats.ttest_1samp(data, 0, nan_policy='omit', axis=0)
    return statistics, pval


def compare_groups(data, group1, group2, testtype='t'):
    """
    Compares the maps of two groups of subjects pixel-wise

    :param data: 3-D data matrix of subject-wise colouring maps. Axis 0 represents subjects.
    :param group1: indices of group1 members in the matrix
    :param group2: indices of group2 members in the matrix
    :param testtype: should the groups be compared using a two sample t-test (default) or z-test of proportions?
    :return: two matrices, with the test statistic and p-value for the comparison per each pixel
    """
    # copy the data for each group to avoid accidentally making edits to original data
    g0_data = np.copy(data[group1])
    g1_data = np.copy(data[group2])
    dims = g0_data.shape
    if testtype=='z':
        # test of proportions
        if np.amin(g0_data) < 0 or np.amin(g1_data) < 0:
            print('test of proportions is only defined for data with no negative values')
            return
        # binarize and count hits
        g0_data = binarize(g0_data)
        g1_data = binarize(g1_data)
        successes = [np.concatenate(np.sum(g0_data, axis=0)), np.concatenate(np.sum(g1_data, axis=0))]
        counts = [np.concatenate(np.nansum(~np.isnan(g0_data), axis=0)),
                  np.concatenate(np.nansum(~np.isnan(g1_data), axis=0))]
        # run proportions test for each pixel based on number of observations(counts) and number of coloured pixels (successes)
        map_out = list(map(lambda x, y: proportions_ztest(x,y), np.transpose(successes), np.transpose(counts))) # a little slow, is there a better iteration?
        statistics_twosamp = np.reshape(np.transpose(map_out)[0], (dims[1],dims[2]))
        pval_twosamp = np.reshape(np.transpose(map_out)[1], (dims[1],dims[2]))
    elif testtype=='t':
        # two sample t-test
        statistics_twosamp, pval_twosamp = stats.ttest_ind(g0_data, g1_data, axis=0, nan_policy='omit')
    else:
        print("acceptable comparison types are 'z' and 't', not sure what you want")
    return statistics_twosamp, pval_twosamp


def correlate_maps(data, corr_with):
    """
    Correlates a set of subject-wise colouring maps with a vector of values (e.g. a background factor)

    :param data: 3-D data matrix of the subject-wise colouring maps. Axis 0 represents subjects.
    :param corr_with: vector of values (1 per subject) to correlate with.
    :return: map with correlation coefficient for each.
    """
    # TODO: move from np.correlate to something which allows spearman and pearson,
    #  like pandas corr or scipy spearmanr and pearsonr? Also get p-values
    dims = data.shape
    if dims[0] is not len(corr_with): # NB: change this to a proper error at some point
        print('You need to provide exactly one value per subject for the analysis. Stopping execution.')
        return np.nan
    # temporarily change data to 2-D to enable correlation analysis
    data_reshaped = np.reshape(data, (dims[0], -1))
    # run correlation on each pixel separately
    corr_res = np.apply_along_axis(np.correlate, 0, data_reshaped, corr_with)
    # reshape result
    corr_map = np.reshape(corr_res, (dims[1], dims[2]))
    return corr_map


def p_adj_maps(pval_map, mask=None, alpha = 0.05, method='fdr_bh'):
    """
    An easier interface to correct p-values for multiple comparisons. By default, implements False Detection Rate
    correction (Benjamini/Hochberg), but multiple other methods can be selected.

    Implements
    https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html

    :param pval_map: a 2-D matrix of p-values
    :param mask: optional. A boolean matrix with the areas inside the body outline set to 1/TRUE. If provided, will
    reduce the number of comparisons to correct for, as only areas inside the body outline (i.e. areas of interest)
    are considered.
    :param alpha: selected alpha-level (default 0.05)
    :param method:
    :return: 2-D matrix of the same size as first parameter, with p-values corrected for multiple comparison
    """
    dims = pval_map.shape
    if mask is None:
        data_reshaped = np.reshape(pval_map, (dims[0], -1))
    else:
        if dims!=mask.shape:
            print('expected mask to be same shape as data, cannot continue')
            return
        else:
            # if we have mask, we can just pick the relevant numbers
            print('found mask of the right size')
            data_reshaped = pval_map[mask.astype(int)>0]
    reject, pvals_corrected, alpacSidak, alhacBonferroni = multipletests(data_reshaped, alpha, method)
    if mask is None:
        pval_map_corrected = np.reshape(pvals_corrected, (dims[1],dims[2]))
    else:
        pval_map_corrected = np.ones(dims)
        pval_map_corrected[mask.astype(int)>0] = pvals_corrected
    return pval_map_corrected


def read_in_mask(file1, file2=None):
    """
    Easily read in a black and white mask image and change to binary numpy array to use in other functions.
    If the data need left and right mask separately, please providethe mask to use on the left-hand side as the
    first argument.

    :param file1: Black-and-white mask image, where black shows areas inside the mask (i.e. to be included) and white
    shows areas outside of the mask (i.e. background).
    :param file2: Mask to use for the right side, if any
    :return: numpy array of the mask with 1=include, 0=exclude
    """
    mask_array = io.imread(file1, as_gray=True)
    mask_array[mask_array < 1] = 0
    mask_array = mask_array * -1
    mask_array = mask_array + 1
    if file2 is not None:
        mask_other_side = io.imread(file2, as_gray=True)
        mask_other_side[mask_other_side < 1] = 0
        mask_other_side = mask_other_side * -1
        mask_other_side = mask_other_side + 1
        mask_array = np.concatenate((mask_array, mask_other_side), axis=1)
    return mask_array


def count_pixels(data, mask=None):
    """
    Count the number and proportion of coloured pixels per subject

    :param data: the 3D-data frame where to count the
    :param mask: optional. If provided, takes
    :return: number of coloured pixels per subject
    """
    data = binarize(data)
    # sum all cells for each subject
    if mask is None:
        counts_vector = np.sum(np.sum(data, axis=1), axis=1)
        n_pixels = data.shape[1]*data.shape[2]
    else:
        inside_mask = data[:,mask==1]
        n_pixels = np.sum(np.sum(mask))
        counts_vector = np.sum(inside_mask, axis=1)
    prop_vector = [x / n_pixels for x in counts_vector]
    return counts_vector, prop_vector
