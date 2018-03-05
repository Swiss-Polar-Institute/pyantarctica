# import os
import numpy as np
import pandas as pd


class ACEdata:
    """Waves and Aerosols dataset:
    Initialization args:
        - root : "data" folder
        - dataset : 'wave' or 'aerosol' (will then be merged somehow)
    """

    # Subfolders
    raw_folder = 'raw'
    processed_folder = 'processed'
    intermediate_folder = 'intermediate'

    def __init__(self, name, data_folder='./data/'):
        self.data_folder = data_folder + '/'
        self.name = name

        if self.name is 'wave':
            self.dataname = 'waveData_Toffoli_all'
            extension = '.txt'
            skiplines = 23
            # nantype = 'nan'
            ncols = 12
            delimiter = '\t'

        elif self.name is 'aerosol':
            self.dataname = 'aerosol_cleaned'#'AerosolData_Schmale_all'
            extension = '.txt'
            skiplines = 0#11
            # nantype = '\t'
            ncols = 3
            delimiter = '\t'

        else:
            print('dataset not handled yet: must be ''wave'' or ''aerosol''')

        self.datatable = self.load(extension, skiplines, delimiter)

    def load(self, ext, skiplines, delim):
        """Load and sets data object named 'dataset' """
        fullfolder = self.data_folder + self.raw_folder + '/' + self.dataname + ext
        datatable = pd.read_csv(fullfolder, skiprows=skiplines-1,
                                delimiter=delim,
                                index_col=None)
        
        # if self.name is 'wave':
        #   print(self.name)
        #    datatable.rename(index=str, columns={"HS_w": "Hs_w", "Tp_W": "Tp_w", "phase_vel_W": "phase_vel_w"}, inplace=True)

        
        print('loaded from ' + fullfolder)
        return datatable

    def convert_time_to_unixts(self, new_column_name):
        """Convert raw timestamp to unix timestamp"""
        from datetime import datetime, tzinfo, timedelta
        #from time import mktime
        from calendar import timegm

        if new_column_name in self.datatable:
            print('Converted time already present')
            return
        
        class UTC(tzinfo):
            """UTC subclass"""

            def utcoffset(self, dt):
                return timedelta(0)

            def tzname(self, dt):
                return "UTC"

            def dst(self, dt):
                return timedelta(0)

        if self.name is 'aerosol':
            self.datatable['t_series_aerosol'] = (self.datatable['Date'] + ' ' + self.datatable['time'])
            datetime_object = [datetime.strptime(str(date), '%d.%m.%y %H:%M:%S') for date in
                               self.datatable['t_series_aerosol']]
        elif self.name is 'wave':
            datetime_object = [datetime.strptime(str(date), '%d.%m.%Y %H:%M') for date in
                               self.datatable['t_series_waves']]

        datetime_obj = [date_.replace(tzinfo=UTC()) for date_ in datetime_object]
        timestamp = [timegm(date_.timetuple()) for date_ in datetime_obj]

        self.datatable[new_column_name] = pd.DataFrame(timestamp)


def feature_expand(table, trans):

    table_new = pd.DataFrame()

    for key in trans:
        print(key)
        if key is 'square':
            for col in table.columns.tolist():
                table_new[col + '_square'] = table[col]**2
        elif key is 'cube':
            for col in table.columns.tolist():
                table_new[col + '_cube'] = table[col]**3

    return table_new














