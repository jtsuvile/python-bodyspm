# imports
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from classdefinitions import Subject, Stimuli
import pickle

def preprocess_subjects(subnums, indataloc, outdataloc, stimuli, bgfiles=None,fieldnames=None):
    '''Reads in data from web interface output and writes the subjects out to .csv files (colouring data) and
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

    Does not return anything, data are stored as files to outdataloc'''

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

def combine_data(dataloc, subnums, save=False):
    '''
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
    '''

    # NB: add sub bg info into all_data
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


def compare_groups(data, group1, group2, testtype='t'):
    '''
    Compares the maps of two groups of subjects pixel-wise

    :param data: 3-D data matrix of the subject-wise colouring maps. Axis 0 represents subjects.
    :param group1: indices of group1 members in the matrix
    :param group2: indices of group2 members in the matrix
    :param testtype: should the groups be compared using a two sample t-test (default) or z-test of proportions?
    :return: two matrices, with the test statistic and p-value for the comparison per each pixel
    '''
    # copy the data for each group to avoid accidentally making edits to original data
    g0_data = np.copy(data[group1])
    g1_data = np.copy(data[group2])
    if testtype=='z':
        # test of proportions
        if np.amin(g0_data) < 0 or np.amin(g1_data) < 0:
            print('test of proportions is only defined for data with no negative values')
            return
        # binarize and count hits
        g0_data[g0_data > 0] = 1
        g1_data[g1_data > 0] = 1
        successes = [np.concatenate(np.sum(g0_data, axis=0)), np.concatenate(np.sum(g1_data, axis=0))]
        counts = [np.concatenate(np.nansum(~np.isnan(g0_data), axis=0)),
                  np.concatenate(np.nansum(~np.isnan(g1_data), axis=0))]
        # run proportions test for each pixel based on number of observations(counts) and number of coloured pixels (successes)
        map_out = list(map(lambda x, y: proportions_ztest(x,y), np.transpose(successes), np.transpose(counts))) # a little slow, is there a better iteration?
        statistics_twosamp = np.transpose(map_out)[0]
        pval_twosamp = np.transpose(map_out)[1]
    elif testtype=='t':
        # two sample t-test
        statistics_twosamp, pval_twosamp = stats.ttest_ind(g0_data, g1_data, axis=0, nan_policy='omit')
    else:
        print("acceptable comparison types are 'z' and 't', not sure what you want")
    return statistics_twosamp, pval_twosamp


def correlate_maps(data, corr_with):
    '''
    Correlates a set of subject-wise colouring maps with a vector of values (e.g. a background factor)

    :param data: 3-D data matrix of the subject-wise colouring maps. Axis 0 represents subjects.
    :param corr_with: vector of values (1 per subject) to correlate with.
    :return: map with correlation coefficient for each.
    '''
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