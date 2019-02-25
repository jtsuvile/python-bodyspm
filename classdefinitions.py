import numpy as np
import os
from cv2 import GaussianBlur
import simplejson as json

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

    def read_data(self, dataloc, data_names, onesided):
        # add : check for length of onesided (either one or same as data_names)
        for stimulus in data_names:
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
                arr_color[x.astype(int), y.astype(int)] = arr_color[x.astype(int), y.astype(int)] + 1
                # add blur to replicate the effect of spray can in the web interface
                # NB: size of blur might need to be changed depending on your data collection settings
                as_coloured = GaussianBlur(arr_color, (0, 0), 8.5)
                # cut out the figure area.
                # NB: depending on the size & positioning of base image on your web interface,
                # you might need to change the indices below to give the best possible fit
                if onesided:
                    raw_res = as_coloured[10:531, 33:203] - as_coloured[10:531, 698:868]  # this creates 522*171 array
                else:
                    raw_res = as_coloured[10:531, 35:205] - as_coloured[10:531, 700:870]  # this creates 522*342 array
                self.add_data(stimulus, {'name': stimulus, 'onesided': onesided, 'rawdata': raw_res})

    def write_data(self, fileloc):
        subdir = fileloc + '/'+str(self.name) +'/'
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        # saving each color map to a separate comma separated file and
        for key, value in self.data.items():
            datafile = subdir + key + '_as_matrix.csv'
            self.data_loc.append(datafile)
            np.savetxt(datafile, value['rawdata'], fmt='%.10f', delimiter=',')
        return "done"

    def write_sub_to_file(self, fileloc):
        self.write_data(fileloc)
        filename = fileloc + '/sub_' + str(self.name) + '_info.json'
        outdata = self.bginfo
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

    def read_sub_from_file(self, fileloc):
        # read in data and other subject info from a json file
        filename = fileloc + '/sub_' + str(self.name) + '_info.json'
        with open(filename) as f:
            indata = json.load(f)
        self.group = indata['group']
        self.data_loc = indata['datafiles']
        print(type(indata))
        for key, value in indata.items():
            if key not in ('name','group','datafiles'):
                self.bginfo[key] = value
        self.data_from_file()

    # TODO : TO ADD
    # rough plot data

    def __str__(self):
        return "subject with id "+str(self.name)+', has '+str(len(self.data.keys()))+' colour maps'
