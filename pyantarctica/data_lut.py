import pandas as pd
import numpy as np
import os
import sys

import pyantarctica.dataset as dataset

class Varlut:

    def __init__(self, table_address, root_folder='.'):
        self.table = pd.read_csv(table_address)
        # print(self.table['key'])
        self.table.set_index(self.table['key'], inplace=True)
        self.table.drop('key', axis=1, inplace=True)
        self.root_folder = root_folder

        if sys.platform == 'darwin':
            # print('deleting .DS_store')
            os.system("find . -name '.DS_Store' -delete")
            self.sep = '/'
        elif sys.platform == 'win32':
            self.sep = "\\"


    def getv_vname(self, vname):
        row = np.where(vname == self.table['variable'])[0]
        if len(row) > 1:
            print('found more than one match, returning the first')
            row = row[0]

        key = self.table.iloc[row,:].name
        print(key)
        fname = self.table.loc[key, 'filepath'] + self.table.loc[key, 'filename']

        t_ = dataset.read_standard_dataframe(self.root_folder + self.sep + fname)
        t_ = t_[vname]
        return t_

    def getv_fname_vname(self, fname, vname):

        row_v = np.where(vname == self.table['variable'])[0]
        row_f = np.where(fname == self.table['filename'])[0]

        # The correct file can only be given by the intersection of the rows
        row = np.intersect1d(row_f, row_v)
        key = self.table.iloc[row[0],:].name

        print(key)
        fname = self.table.loc[key, 'filepath'] + self.table.loc[key, 'filename']

        t_ = dataset.read_standard_dataframe(self.root_folder + self.sep + fname)
        t_ = t_[vname]
        return t_
