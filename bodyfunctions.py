# imports
import numpy as np
import pandas as pd
import os
from scipy import stats
from skimage import io
from statsmodels.stats.proportion import proportions_ztest, proportions_chisquare
from statsmodels.stats.multitest import multipletests
from classdefinitions import Subject, Stimuli
from datetime import datetime
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt


## Data wrangling
# functions for getting data into the correct format for other analyses
# Specifically designed to work with the Aalto University emBODY system


def preprocess_subjects(subnums, indataloc, outdataloc, stimuli, bgfiles=None, fieldnames=None,
                        intentionally_empty = False):
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
        sub.read_data(indataloc, stimuli, False, intentionally_empty)
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


def add_background_table(new_bgdata, linking_col, subloc, exclude=[], override=True):
    """
    Add new background information from a data frame to a subject after preprocessing.
    Adding is done based on a linking column (subject identifier).
    The names of the background variables are taken from columns of the data frame.

    :param new_bgdata: table with the new background information
    :param linking_col: column name for the column containing subject identifiers
    :param subloc: path to where the preprocessed subject data files are stored
    :param exclude: optional. column names that should be ignored (not added to the subject)
    :param override: optional. if the subject already has a background item with the same name, should this be overriden by the new data?

    """
    existing_subs = os.listdir(subloc)
    group_colnames = new_bgdata.columns.tolist()
    if linking_col in group_colnames:
        group_colnames.remove(linking_col)
    else:
        return("cannot find linking identifier ", linking_col)
    if exclude:
        for colname in exclude:
            group_colnames.remove(colname)
    for subject in new_bgdata[linking_col]:
        if str(subject) in existing_subs:
            temp_sub = Subject(subject)
            temp_sub.read_sub_from_file(subloc, noImages=True)
            for column in group_colnames:
                if column not in temp_sub.has_background() or override==True:
                    temp_sub.add_background(column, new_bgdata.loc[new_bgdata[linking_col] == subject][column].values[0])
                    temp_sub.write_sub_to_file(subloc)
                else:
                    print("not updating subject ", subject, " for ", column)
        else:
            print('no subject ', subject, 'found')
    return "done"


def combine_data(dataloc, subnums, groups=None, save=False, noImages = False):
    """
    Combines a data set from subjects who have been written to file.

    :param dataloc: where the subject data files have been saved. Assumes .json files for subjects and
    stimuli are located in this folder
    :param subnums: which subjects to combine (list). Assumes one .json file per subject
    :param groups: list of group definitions to be included, must be same length and in same order as subnums
    :param save: do you want the combined data set saved into file (pickled)? (Boolean)
    :param noImages: do you want to just combine background data? (useful for extremely large data sets)
    :return: combined data for the defined subjects.
    The data are stored in a dictionary, where each body map will be presented as N * X * Y numpy array,
    where N is length of subnums, and X and Y are the dimensions of the maps. The dictionary will have keys for
    each stimulus (with the value being 3-D Numpy array) and 'subids' (list of the subids in the same order as they appear
    in the data arrays).
    """

    # NB: replication of code for saving with/without images. TODO: edit to remove repeat code

    filename = dataloc + '/dataset_' + datetime.now().strftime("%d%m%Y-%H%M") + '.h5'
    stim = Stimuli(fileloc=dataloc, from_file=True)
    size_onesided = (522, 171)
    size_twosided = (522, 342)
    all_res = {}
    all_res['bg'] = pd.DataFrame(index=subnums)
    all_res['bg']['subid'] = np.nan
    all_res['stimuli'] = stim
    # if groups is not None and np.shape(groups)==np.shape(subnums):
    #     print('added group definitions')
    #     all_res['bg']['groups'] = groups
    # first attempt with H5
    have_written_bg = False
    if noImages:
        for j, subnum in tqdm(enumerate(subnums), desc="subjects"):
            temp_sub = Subject(subnum)
            print(temp_sub)
            temp_sub.read_sub_from_file(dataloc, noImages)
            if sum(all_res['bg']['subid'] == subnum) == 0:
                all_res['bg'].loc[subnum, 'subid'] = int(subnum)
                for bgkey, bgvalue in temp_sub.bginfo.items():
                    #if bgkey != 'profession':  # cannot be neatly converted to numeric, excluding for now
                    if not isinstance(bgvalue, str):
                        if not bgkey in all_res['bg'].columns:
                            all_res['bg'][bgkey] = np.nan
                        all_res['bg'].loc[subnum, bgkey] = int(bgvalue)
        if save:
            with h5py.File(filename, 'a') as store:
                if not have_written_bg:
                    print('writing out background data')
                    if groups is not None:
                        dt = h5py.special_dtype(vlen=str)
                        store.create_dataset('groups', data=groups, dtype=dt)
                    for bgkey, bgvalue in all_res['bg'].items():
                        store.create_dataset(bgkey, data=bgvalue.to_numpy(), dtype="int32")
                    have_written_bg = True
    else:
        for key in tqdm(stim.all.keys(), desc="bodymap number"):
            #print(key)
            if stim.all[key]['onesided']:
                data_matrix = np.zeros((len(subnums), size_onesided[0], size_onesided[1]))
            else:
                data_matrix = np.zeros((len(subnums), size_twosided[0], size_twosided[1]))
            for j, subnum in tqdm(enumerate(subnums), desc="subjects"):
                #print(subnum)
                temp_sub = Subject(subnum)
                temp_sub.read_sub_from_file(dataloc, noImages)
                data_matrix[j] = temp_sub.data[key]
                if sum(all_res['bg']['subid'] == subnum) == 0:
                    all_res['bg'].loc[subnum, 'subid'] = int(subnum)
                    for bgkey, bgvalue in temp_sub.bginfo.items():
                        try:  # some variables cannot be neatly converted to numeric, excluding for now
                            float(bgvalue)
                            if not bgkey in all_res['bg'].columns:
                                all_res['bg'][bgkey] = np.nan
                            all_res['bg'].loc[subnum, bgkey] = int(bgvalue)
                        except ValueError:
                            pass
            if save:
                with h5py.File(filename, 'a') as store:
                    #print('writing out ', key)
                    store.create_dataset(key, data=data_matrix)
                    if not have_written_bg:
                        print('writing out background data')
                        if groups is not None:
                            dt = h5py.special_dtype(vlen=str)
                            store.create_dataset('groups', data=groups, dtype=dt)
                        for bgkey, bgvalue in all_res['bg'].items():
                            store.create_dataset(bgkey, data=bgvalue.to_numpy(), dtype="int32")
                        have_written_bg = True
    #print("combined all data successfully ")
    return all_res


# Functions used in analyses
# Input is always a data matrix


def binarize(data, threshold=0.007):
    """
    Change data from colouring (with blur) binary(ish) coloured / not coloured format.
    If the data have only positive values, the returned map is truly binary (0/1).
    If the map also has negative values (e.g. emotion maps with activations and inactivations), the
    returned map can have three values: 0, 1, and -1.

    NB: Default threshold 0.007 chosen as limit based on what limit replicates coloring best in Aalto system (March 2019).
    The best value for this parameter will depend on brush & blur settings.
    If changed, you should definitely do a visual inspection of the result against known colouring.

    :param data: matrix with colouring data
    :return: same matrix, with coloured areas changed to 1 and non-coloured changed to 0
    """
    data[data > threshold] = 1
    data[(data <= threshold) & (data >= -threshold)] = 0
    data[data < -threshold] = -1
    return data


def count_pixels(data, mask=None, threshold=0.007):
    """
    Count the number and proportion of coloured pixels per subject

    :param data: the 3D-data frame where to count the
    :param mask: optional. If provided, takes values inside mask into account in counting proportion colored
    :return: number of coloured pixels per subject
    """
    data = binarize(data, threshold)
    data = np.exp2(data)
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


def count_pixels_posneg(data, mask=None, threshold=0.007):
    """
    Count the number and proportion of coloured pixels per subject

    :param data: matrix with the colouring data to be counted
    :param mask: optional. If provided, takes values inside mask into account in counting proportion colored
    :return: number of coloured pixels per subject
    """
    data = binarize(data, threshold)
    data_neg = data.copy()
    data_neg[data_neg > 0] = 0
    data_neg = data_neg * -1
    data_pos = data.copy()
    data_pos[data_pos < 0] = 0
    # sum all cells for each subject
    if mask is None:
        mask = np.ones((data.shape[1],data.shape[2]))
    else:
        mask[mask != 1] = 0
    inside_mask_pos = data_pos[:, mask == 1]
    pos_vector = np.sum(inside_mask_pos, axis=1)
    inside_mask_neg = data_neg[:, mask == 1]
    neg_vector = np.sum(inside_mask_neg, axis=1)
    n_pixels = np.sum(np.sum(mask))
    prop_pos = np.array([x / n_pixels for x in pos_vector])
    prop_neg = np.array([x / n_pixels for x in neg_vector])

    return pos_vector, prop_pos, neg_vector, prop_neg


def one_sample_t_test(data):
    """
    one sample t-test to see if coloured data is significantly more than 0

    :param data: a 3-D matrix of subject-wise colouring maps
    :return: statistics: t statistic for each pixel
    :return: p-value for each pixel (uncorrected)
    """
    statistics, pval = stats.ttest_1samp(data, 0, nan_policy='omit', axis=0)
    return statistics, pval


def compare_groups(group1, group2, testtype='t'):
    """
    Compares the maps of two groups of subjects pixel-wise

    :param data:
    :param group1: Data for group 1. 3-D data matrix of subject-wise colouring maps. Axis 0 represents subjects.
    :param group2: Data for group 2. 3-D data matrix of subject-wise colouring maps. Axis 0 represents subjects.
    :param testtype: should the groups be compared using a two sample t-test (default) or z-test of proportions?
    :return: two matrices, with the test statistic and p-value for the comparison per each pixel
    """
    # copy the data for each group to avoid accidentally making edits to original data
    g0_data = np.copy(group1)
    g1_data = np.copy(group2)
    dims = g0_data.shape
    if testtype=='z':
        # test of proportions
        if np.amin(g0_data) < 0 or np.amin(g1_data) < 0:
            print('test of proportions is only defined for data with no negative values')
            return
        # binarize and count hits
        g0_data = binarize(g0_data)
        g1_data = binarize(g1_data)
        successes = [np.concatenate(np.nansum(g0_data, axis=0)), np.concatenate(np.nansum(g1_data, axis=0))]
        counts = [np.concatenate(np.nansum(~np.isnan(g0_data), axis=0)),
                  np.concatenate(np.nansum(~np.isnan(g1_data), axis=0))]
        # tried an ugly solution to division by 0 in cases of extreme difference
        successes[0] = successes[0]+1
        successes[1] = successes[1]+1

        # run proportions test for each pixel based on number of observations(counts) and number of coloured pixels (successes)
        map_out = list(map(lambda x, y: proportions_ztest(x,y), np.transpose(successes), np.transpose(counts))) # a little slow, is there a better iteration?
        #map_out = list(map(lambda x, y: proportions_chisquare(x,y), np.transpose(successes), np.transpose(counts))) # a little slow, is there a better iteration?
        statistics_twosamp = np.reshape(np.transpose(map_out)[0], (dims[1],dims[2]))
        pval_twosamp = np.reshape(np.transpose(map_out)[1], (dims[1],dims[2]))
    elif testtype=='t':
        # two sample t-test
        statistics_twosamp, pval_twosamp = stats.ttest_ind(g0_data, g1_data, axis=0, nan_policy='omit')
    else:
        print("acceptable comparison types are 'z' and 't', not sure what you want")
    return statistics_twosamp, pval_twosamp


def correlate_maps(data, corr_with, method):
    """
    Correlates a set of subject-wise colouring maps with a vector of values (e.g. a background factor)

    :param data: 3-D data matrix of the subject-wise colouring maps. Axis 0 represents subjects.
    :param corr_with: vector of values (1 per subject) to correlate with.
    :return: map with correlation coefficient for each and map with p-values
    """
    dims = data.shape
    if dims[0] is not len(corr_with): # NB: change this to a proper error at some point
        print('You need to provide exactly one value per subject for the analysis. Stopping execution.')
        return np.nan

    # special implementation for pointbiserialr
    if method == 'pointbiserialr':
        corr_map, p_map = count_pointserialr_for_array(data, corr_with)
        return corr_map, p_map

    # other methods

    # temporarily change data to 2-D to enable vectorized analysis
    data_reshaped = np.reshape(data, (dims[0], -1))
    # run correlation on each pixel separately
    if method == 'spearmanr':
        corr_res, pvals = np.apply_along_axis(stats.spearmanr, 0,
                                              data_reshaped, corr_with,
                                              nan_policy='omit')
    elif method == 'pearson':
        corr_res, pvals = np.apply_along_axis(stats.pearsonr, 0,
                                              data_reshaped, corr_with,
                                              nan_policy='omit')
    else:
        raise ValueError(f"analysis type {method} is not implemented. Or maybe there's a typo?")

    # reshape result
    corr_map = np.reshape(corr_res, (dims[1], dims[2]))
    p_map = np.reshape(pvals, (dims[1], dims[2]))
    return corr_map, p_map


# Helper functions


def get_latest_datafile(datadir):
    """
    :param datadir: where to search for datasets
    :return: full path to latest datafile named dataset_ in the given datadir
    """
    latestfile = ''
    for file in os.listdir(datadir):
        if file.startswith("dataset"):
            if latestfile == '' or os.path.getmtime(datadir +  '/' + file) > os.path.getmtime(datadir + '/' + latestfile):
                latestfile = file
            dataloc = os.path.join(datadir, latestfile)
    return dataloc


def make_qc_figures(subnums, indataloc, stimuli, outdataloc = None):
    """
    Draw the entire colouring area for the given subjects. This is particularly useful in colouring quality control.
    :param subnums: list of subject id's whose data to inspect
    :param indataloc: where to look for preprocessed subject data
    :param stimuli:
    :param outdataloc: where to save the figures (defaults to same as above)

    """
    if outdataloc is None:
        outdataloc = indataloc

    for i, subnum in enumerate(subnums):
        print("making qc figures for subject " + str(subnum) + " which is " + str(i + 1) + "/" + str(len(subnums)))
        sub = Subject(subnum)
        sub.read_data(indataloc, stimuli, whole_image=True)
        sub.draw_sub_data(stimuli, fileloc=outdataloc, qc=True)
    return "done with qc figures"


def make_correlation_plot(result_map_r, result_map_p, mask, suptitle, outdataloc):
    # fix plot color definitions
    hot = plt.cm.get_cmap('hot', 256)
    new_cols = hot(np.linspace(0, 1, 256))

    cold = np.hstack((np.fliplr(new_cols[:, 0:3]), new_cols[:, 3][:, None]))
    newcolors = np.vstack((np.flipud(cold), new_cols))
    newcolors = np.delete(newcolors, np.arange(200, 312, 2), 0)

    cmap = 'coolwarm'
    vmin = -1
    vmax = 1

    # make versions of array with and without fdr correction
    result_map_r_no_fdr = result_map_r.copy()
    result_map_r_with_fdr = result_map_r.copy()

    result_map_r_no_fdr[result_map_p > 0.05] = 0
    masked_result_map_no_fdr = np.ma.masked_where(mask != 1, result_map_r_no_fdr)

    result_map_p_corrected, twosamp_reject = p_adj_maps(result_map_p, mask=mask, method='fdr_bh')
    result_map_r_with_fdr[result_map_p_corrected > 0.05] = 0
    result_map_r_with_fdr[np.isnan(result_map_p_corrected)] = 0

    masked_result_map_with_fdr = np.ma.masked_where(mask != 1, result_map_r_with_fdr)

    # plotting
    fig = plt.figure(figsize=(20, 10))

    ax1 = plt.subplot(131)
    img1 = plt.imshow(result_map_r, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.title.set_text("Correlation coefficient, also showing insignificant")
    fig.colorbar(img1, fraction=0.046, pad=0.04)
    ax1.axis('off')

    ax2 = plt.subplot(132)
    img2 = plt.imshow(masked_result_map_no_fdr, cmap=cmap, vmin=-1, vmax=1)
    ax2.title.set_text('Significant correlations before FDR')
    fig.colorbar(img2, fraction=0.046, pad=0.04)
    ax2.axis('off')

    ax3 = plt.subplot(133)
    img3 = plt.imshow(masked_result_map_with_fdr, cmap=cmap, vmin=-1, vmax=1)
    ax3.title.set_text('Correlations that survive FDR')
    fig.colorbar(img3, fraction=0.046, pad=0.04)
    ax3.axis('off')

    fig.suptitle(suptitle)

    plt.savefig(outdataloc + '/' + suptitle + '.png')
    plt.close()

    return "done with plotting"


def read_in_mask(file1, file2=None):
    """
    Easily read in a black and white mask image and change to binary numpy array to use in other functions.
    If the data need left and right mask separately, please provide the mask to use on the left-hand side as the
    first argument.

    :param file1: Black-and-white mask image, where black shows areas inside the mask (i.e. to be included) and white
    shows areas outside of the mask (i.e. background).
    :param file2: Mask to use for the right side, if any
    :return: numpy array of the mask with 1=include, 0=exclude
    """
    mask_array = io.imread(file1)
    dims = mask_array.shape
    if len(dims) == 3:
        mask_array = mask_array[:, :, 0]
    mask_array[mask_array < 1] = 0
    mask_array = mask_array * -1
    mask_array = mask_array + 1
    if file2 is not None:
        mask_other_side = io.imread(file2)
        dims = mask_other_side.shape
        if len(dims) == 3:
            mask_other_side = mask_other_side[:, :, 0]
        mask_other_side[mask_other_side < 1] = 0
        mask_other_side = mask_other_side * -1
        mask_other_side = mask_other_side + 1
        mask_array = np.concatenate((mask_array, mask_other_side), axis=1)
    return mask_array


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
        data_reshaped = np.reshape(pval_map, (-1, 1)).flatten()
        #print(dims)
        #print(data_reshaped.shape)
    else:
        if dims != mask.shape:
            print('expected mask to be same shape as data, cannot continue')
            return
        else:
            # if we have mask, we can just pick the relevant numbers
            #print('found mask of the right size')
            data_reshaped = pval_map[mask.astype(int)>0]
    reject, pvals_corrected, alpacSidak, alhacBonferroni = multipletests(data_reshaped, alpha, method)
    if mask is None:
        pval_map_corrected = np.reshape(pvals_corrected, (dims[0], -1))
        reject_map = np.reshape(reject, (dims[0],-1))
    else:
        pval_map_corrected = np.ones(dims)
        pval_map_corrected[mask.astype(int)>0] = pvals_corrected
        reject_map = np.ones(dims)
        reject_map[mask.astype(int) > 0] = reject
    return pval_map_corrected, reject_map


def align_data(sublist, subarray, dataframe, subidcolumn='subid'):
    """ Aligns two data items: a matrix and a dataframe in the same order
    Sublist: a list of subject id's. This MUST be in the same order as the subject dimension in subarray
    Subarray: an array with embody data. Subject dimension MUST be the first dimension.
    Dataframe: a data frame with additional subject data to be aligned to the subarray
    Subidcolumn: if the subject id column in the dataframe is not 'subid', define the column name
    """

    # First, let's create a mapping from subject ID to index in data_array
    subject_to_index = {subject: idx for idx, subject in enumerate(sublist)}

    # Create a new array to store the aligned data
    # We'll first identify which subjects in df are also in subs
    common_subjects = [s for s in dataframe[subidcolumn].values if s in subject_to_index]
    n_common = len(common_subjects)

    # Get the shape of each "picture" in data_array
    pic_shape = subarray.shape[1:]  # This will be (height, width)

    # Create new array with the right size
    aligned_data = np.zeros((n_common,) + pic_shape)

    # Create a new dataframe with only the common subjects
    aligned_df = dataframe[dataframe[subidcolumn].isin(common_subjects)].copy().reset_index(drop=True)

    # Fill the aligned_data array with data from data_array in the order of aligned_df
    for i, subject in enumerate(aligned_df[subidcolumn]):
        original_idx = subject_to_index[subject]
        aligned_data[i] = subarray[original_idx]

    return aligned_data, aligned_df


def count_pointserialr_for_array(data, corr_with):
    """
    Helper function to count pointserialr correlation. 
    """
    # NB: The function is sensitive to missing values which is why this is not implemented with 
    # np.apply_along_axis
    dim1 = data.shape[1]
    dim2 = data.shape[2]

    result_map_r = np.zeros([dim1, dim2])
    result_map_p = np.zeros([dim1, dim2])

    # are we missing anything in the corr variable?
    missing_corrvar = np.isnan(corr_with)
    if (sum(missing_corrvar > 0)):
        problem_indices_corrvariable = np.argwhere(missing_corrvar)
    else:
        problem_indices_corrvariable = np.array([])

    for ind_i in range(0, dim1):
        for ind_j in range(0, dim2):
            # remove variables with nans
            problem_indices_bodymap = np.argwhere(np.isnan(data[:, ind_i, ind_j]))
            all_problem_indices = list(set(problem_indices_corrvariable.flatten()) | set(problem_indices_bodymap.flatten()))

            # drop problematic values
            bodymap_bin_no_nans = np.delete(data[:, ind_i, ind_j], all_problem_indices)
            corr_variable_no_nans = np.delete(corr_with, all_problem_indices)

            if len(set(bodymap_bin_no_nans)) == 1 | len(set(corr_variable_no_nans)) == 1:
                correlation = 0
                p_value = 1
            else:
                correlation, p_value = stats.pointbiserialr(bodymap_bin_no_nans, corr_variable_no_nans)

            result_map_p[ind_i, ind_j] = p_value
            result_map_r[ind_i, ind_j] = correlation

    return result_map_r, result_map_p
