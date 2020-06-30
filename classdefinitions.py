import numpy as np
import os
from cv2 import GaussianBlur
import simplejson as json
import matplotlib.pyplot as plt
import math


class Stimuli:
    # class to keep the relevant information of each stimulus together
    def __init__(self, names=None, onesided=True, show_names=None, fileloc='', from_file=False):
        self.all = {}
        self.has_show_names = False
        if not from_file:
            if not names:
                print("Need stimulus names")
                return
            # if onesided is provided as just one value, make vector
            if isinstance(onesided, bool):
                new_onesided = np.repeat(onesided, len(names))
            elif len(onesided) == len(names):
                new_onesided = onesided
            else:
                print("need argument 'onesided' either as single boolean value "
                      "or vector with same length as stimulus names")
                return
            for i, name in enumerate(names):
                self.all[name] = {'onesided': new_onesided[i]}
                if show_names and show_names[i]:
                    self.all[name]['show_name'] = show_names[i]
                    self.has_show_names = True
        # NB: surely there is a more elegant way to read stimuli from file?
        else:
            filename = fileloc + '/stimuli_info.json'
            with open(filename) as f:
                indata = json.load(f)
            for key, value in indata.items():
                self.all[key] = value

    def has_show_names(self):
        return self.has_show_names

    def write_stim_to_file(self, fileloc):
        filename = fileloc + '/stimuli_info.json'
        with open(filename, 'w') as json_file:
            json.dump(self.all, json_file)

    def __str__(self):
        if(self.has_show_names):
            return "Stimulus set with "+ str(len(self.all)) + " stimuli defined: " + (', '.join({value['show_name'] for value in self.all.values()}))
        else:
            return "Stimulus set with "+ str(len(self.all)) + " stimuli defined: " + (', '.join(self.all.keys()))


class Subject:

    def __init__(self, name):
        self.name = name
        self.bginfo = {}
        self.group = None
        self.data = {}
        self.data_loc = []

    def add_background(self, bgkey, bgvalue):
        self.bginfo[bgkey] = bgvalue

    def add_data(self, dataname, datavalue):
        self.data[dataname] = datavalue

    def has_data(self):
        return self.data.keys()

    def has_background(self):
        return self.bginfo.keys()

    def read_data(self, dataloc, stim, whole_image = False):
        for stimulus in stim.all.keys():
            data_in = dataloc + '/' + str(self.name) + '/' + stimulus + '.csv'
            if os.path.isfile(data_in):
                # read in data
                csvdata = np.genfromtxt(data_in, delimiter=',')
                split_ind = np.where(csvdata[:, 0] == -1)
                (mouse, paint, mousedown, mouseup) = [csvdata[0:split_ind[0][0], :],
                                                      csvdata[split_ind[0][0] + 1:split_ind[0][1], :],
                                                      csvdata[split_ind[0][1] + 1:split_ind[0][2], :],
                                                      csvdata[split_ind[0][2] + 1:, :]]
                # transfer data from a list of indices to array
                arr_color = np.zeros((600, 900))
                (x, y) = [paint[:, 2], paint[:, 1]]
                x[x <=0] = 0
                x[x >= 600] = 599
                y[y <=0] = 0
                y[y >= 900] = 899
                arr_color[x.astype(int), y.astype(int)] = arr_color[x.astype(int), y.astype(int)] + 1
                # add blur to replicate the effect of spray can in the web interface
                # NB: size of blur might need to be changed depending on your data collection settings
                as_coloured = GaussianBlur(arr_color, (0, 0), 8.5)
                # cut out the figure area.
                # NB: depending on the size & positioning of base image on your web interface,
                # you might need to change the indices below to give the best possible fit
                if whole_image:
                    raw_res = as_coloured # show the entire painting surface, great for QC
                else:
                    if stim.all[stimulus]['onesided']:
                        raw_res = as_coloured[9:531, 32:203] - as_coloured[9:531, 697:868]  # this creates 522*171 array
                    else:
                        raw_res = np.hstack((as_coloured[9:531, 34:205], as_coloured[9:531, 699:870]))  # this creates 522*342 array
                    # Quality control step to check if subject has filled in intentionally left empty
                    # TODO: add flag to disable for studies that don't have this!
                    # TODO: test on larger dataset when server is back up!
                    if np.count_nonzero(raw_res) == 0 and not self.map_intentionally_empty(as_coloured):
                        raw_res[:] = np.nan
                self.add_data(stimulus, raw_res)
            else:
                raise IOError("File", data_in, "not found")

    def read_bg(self, dataloc, bgfile, fieldnames):
        # BN: code currently assumes that bgfile is a list and fieldnames is list of lists to accommodate situation
        # where subject fills in a lot of background information. Maybe edit to accept both single value + list and
        # current list + list of lists?
        data_in = dataloc + '/' + str(self.name) + '/' + bgfile
        if os.path.isfile(data_in):
            with open(data_in) as d:
                tempdata = d.readlines()
            templist = tempdata[-1].strip().split(',')
            for i in range(len(fieldnames)):
                self.add_background(fieldnames[i],templist[i])
        else:
            print('cannot find file ', data_in)

    def write_data(self, fileloc):
        subdir = fileloc + '/'+str(self.name) +'/'
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        # saving each color map to a separate comma separated file and
        for key, value in self.data.items():
            datafile = subdir + key + '_as_matrix.csv'
            self.data_loc.append(datafile)
            np.savetxt(datafile, value, fmt='%.10f', delimiter=',')
        return "done"

    def write_sub_to_file(self, fileloc):
        self.write_data(fileloc)
        filename = fileloc + '/sub_' + str(self.name) + '_info.json'
        outdata = self.bginfo.copy()
        outdata['name'] = self.name
        outdata['group'] = self.group
        outdata['datafiles'] = self.data_loc
        with open(filename, 'w') as json_file:
            json.dump(outdata, json_file)

    def data_from_file(self):
        # just read in matrix representation of data
        if len(self.data_loc)==0:
            return "could not find saved data files for subject " + str(self.name)
        else:
            for file in self.data_loc:
                filename = file.split('/')[-1]
                data_key = filename.split('_as_matrix')[0]
                self.data[data_key] = np.loadtxt(file, delimiter=',')

    def read_sub_from_file(self, fileloc, noImages=False):
        # subject info from a json file
        filename = fileloc + '/sub_' + str(self.name) + '_info.json'
        with open(filename) as f:
            indata = json.load(f)
        self.group = indata['group']
        self.data_loc = indata['datafiles']
        for key, value in indata.items():
            if key not in ('name','group','datafiles'):
                self.bginfo[key] = value
        # read in data from the locations specified in the json
        if not noImages:
            self.data_from_file()

    def draw_sub_data(self, stim, fileloc=None, qc=False):
        # edit colormaps in twosided [0,1] and twosided [-1,1] cases
        # make sure non coloured values are white in twosided datas
        twosided_cmap = plt.get_cmap('Greens')
        twosided_cmap.set_under('white', 1.0)
        onesided_cmap = plt.get_cmap('RdBu_r')
        # define grey color to show nan's
        twosided_cmap.set_bad('grey', 0.8)
        onesided_cmap.set_bad('grey', 0.8)
        print(stim.all)
        # find out if each data item is one or twosided
        if qc:
            fig, axes = plt.subplots(figsize=(24, 10), ncols=math.ceil(len(stim.all.keys())/2), nrows=2)
        else:
            all_onesided = [stim.all[key]['onesided'] for key in stim.all.keys()]
            widths = []
            for side in all_onesided:
                if side:
                    widths.append(1)
                else:
                    widths.append(2)
            fig, axes = plt.subplots(figsize=(24, 3), ncols=len(stim.all.keys()), gridspec_kw={'width_ratios': widths})
        for i, key in enumerate(stim.all.keys()):
            print(key)
            if i < math.ceil(len(stim.all.keys()) / 2):
                row = 0
                col = i
            else:
                row = 1
                col = i - math.ceil(len(stim.all.keys()) / 2)
            print(row, col)
            is_onesided = stim.all[key]['onesided']
            value = self.data[key]
            if is_onesided and not qc:
                map_to_plot = np.ma.masked_where(np.isnan(value), value)
                img = axes[i].imshow(map_to_plot, cmap=onesided_cmap, vmin=-0.05, vmax=0.05)
                fig.colorbar(img, ax=axes[i])
            else:
                map_to_plot = np.ma.masked_where(np.isnan(value), value)
                img = axes[row,col].imshow(map_to_plot, cmap=twosided_cmap, vmin=np.finfo(float).eps, vmax=0.05)
                fig.colorbar(img, ax=axes[row,col], fraction=0.04, pad=0.04)
            if 'show_name' in stim.all[key]:
                axes[row, col].set_title(stim.all[key]['show_name'])
            else:
                axes[row, col].set_title(key)
        fig.suptitle("subject : " + self.name)
        fig.tight_layout()
        if fileloc:
            fileloc_fig = fileloc + '/figures/'
            if not os.path.exists(fileloc_fig):
                os.makedirs(fileloc_fig)
            filename = fileloc_fig+ '/sub_' + str(self.name) + '_data.png'
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()

    def map_intentionally_empty(self, array):
        area = array[530:580, 430:480] # this location is specific to pain patients!
        n_nonzero = np.count_nonzero(area)
        n_area = area.size
        if n_nonzero > 0.1 * n_area:
            return True
        else:
            return False

    def __str__(self):
        return "subject with id "+str(self.name)+', has '+str(len(self.data.keys()))+' colour maps'
