#
# Copyright 2017-2018 - Swiss Data Science Center (SDSC)
# A partnership between École Polytechnique Fédérale de Lausanne (EPFL) and
# Eidgenössische Technische Hochschule Zürich (ETHZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

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

    def __init__(self, name, data_folder='./data/', save_data=False):
        self.data_folder = data_folder + '/'
        self.name = name

        col_name = None
        # row reading is 0-indexed (0 = row number 1)
        if self.name is 'wave_old':
            self.dataname = 'waveData_Toffoli_all'
            extension = '.txt'
            column_head = 22
            body = 23
            nantype = 'nan'
            ncols = 12
            delimiter = '\t'
            self.fullfolder = self.data_folder + self.raw_folder + '/' + self.dataname + extension

        elif self.name is 'wave':
            self.dataname = 'wavedata-14.03-fromMat'
            extension = '.csv'
            column_head = 0
            body = 1
            nantype = 'NaN'
            ncols = 13
            delimiter = ','
            self.fullfolder = self.data_folder + self.intermediate_folder + '/' + self.dataname + extension

        elif self.name is 'aerosol':
            self.dataname = 'AerosolData_Schmale_all'
            extension = '.txt'
            column_head = 10
            body = 11#146
            nantype = ''
            delimiter = '\s+'
            self.fullfolder = self.data_folder + self.dataname + extension
            print(self.fullfolder)
        elif self.name is 'windspeed_metstation':
            self.dataname = 'windspeed_metstation'
            extension = '.csv'
            column_head = 4
            body = 5
            nantype = ''
            delimiter = ';'
            self.fullfolder = self.data_folder + self.raw_folder + '/' + self.dataname + extension

        elif self.name is 'windspeed_metstation_corrected':
            self.dataname = 'windspeed_metstation_corrected'
            extension = '.csv'
            column_head = 0
            body = 1
            nantype = ''
            delimiter = ','
            self.fullfolder = self.data_folder + self.intermediate_folder + '/' + self.dataname + extension

        elif self.name is 'particle_size_distribution':
            #self.dataname = '03_smallparticlesizedistribution'
            #extension = '.csv'
            #column_head = 6
            #body = 7
            #col_name=[str(x) for x in range(1,101,1)]
            #nantype = ''
            #delimiter = '\t'
            #self.fullfolder = self.data_folder + self.intermediate_folder + '/' + self.dataname + extension
            self.dataname = '03_smallparticlesizedistriubtion'
            extension = '.txt'
            column_head = None
            body = 6
            col_name=[str(x) for x in range(1,101,1)]
            nantype = ''
            delimiter = '\t'
            self.fullfolder = self.data_folder + self.raw_folder + '/' + self.dataname + extension

        elif self.name is 'meteo_variable_track':
            self.dataname = 'Meteorological_variables_track'
            extension = '.csv'
            column_head = 0
            body=1
            nantype=''
            delimiter=','
            self.fullfolder = self.data_folder + self.intermediate_folder + '/' + self.dataname + extension

        elif self.name is 'meteo_ecmwf_intpol':
            self.dataname = 'ecmwf_intpol_all'
            extension = '.csv'
            column_head = 0
            body=1
            nantype=''
            delimiter=','
            self.fullfolder = self.data_folder + self.raw_folder + '/' + self.dataname + extension

        elif self.name is 'sorpasso_variable_track':
            self.dataname = 'sorpasso_bs_bio'
            extension = '.csv'
            column_head = 0
            body=1
            nantype=''
            delimiter=','
            self.fullfolder = self.data_folder + self.raw_folder + '/' + self.dataname + extension

        elif self.name is 'meteo_water_vapour':
            self.dataname = 'METEO_watervapour_isotopes_5min_v1'
            extension = '.csv'
            column_head = 0
            body=1
            nantype=''
            delimiter=','
            self.fullfolder = self.data_folder + self.raw_folder + '/' + self.dataname + extension

        else:
            print('dataset not handled yet.')

        self.datatable = self.load(column_head, body, delimiter, nantype, col_name)
        self.set_datetime_index()

        if save_data:
            self.datatable.to_csv(self.data_folder + self.intermediate_folder + '/' + self.name + '_postprocessed.csv', sep=',', na_rep='')

##############################################################################################################
    def load(self, column_head, body, delim, nantype, nam=None):
        """Load and sets data object named 'dataset' """
        # fullfolder = self.data_folder + self.raw_folder + '/' + self.dataname + ext

        if column_head is None:
            skipr=range(0,body)
        else:
            skipr=range(column_head+1,body)


        datatable = pd.read_table(self.fullfolder, skip_blank_lines=False, header=column_head,
                                na_values=nantype,
                                delimiter=delim,
                                skiprows=skipr,
                                index_col=False,
                                names=nam)# 

        datatable.columns = [str(x).lower() for x in datatable.columns.tolist()]

        if self.name is 'aerosol':
            # print('fixing columns...')
            inds = datatable['time'].str.contains('NaN', na=True)
            datatable.loc[inds, 'time'] = datatable.loc[inds, 'date']
            datatable.loc[inds, 'date'] = datatable.loc[inds, 'num_conc']
            datatable.loc[inds, 'num_conc'] = np.nan
            datatable['num_conc'] = pd.to_numeric(datatable['num_conc'])

        elif self.name is 'wave':
            datatable.rename(columns={"date": "t_series_waves"}, inplace=True)

        elif self.name is 'windspeed_metstation':
            datatable.rename(columns={"windspeed (m/s)": "wind_metsta"}, inplace=True)

        elif self.name is 'sorpasso_variable_track':
            datatable = pd.read_csv(self.fullfolder)
            datatable.columns = [str(x).lower() for x in datatable.columns.tolist()]

        print('Data successfully loaded from ' + self.fullfolder)

        return datatable

##############################################################################################################
    def convert_time_to_unixts(self, new_column_name):
        """Convert raw timestamp to unix timestamp"""
        from datetime import datetime, tzinfo, timedelta
        #from time import mktime
        from calendar import timegm

        class UTC(tzinfo):
            """UTC subclass"""
            def utcoffset(self, dt):
                return timedelta(0)
            def tzname(self, dt):
                return "UTC"
            def dst(self, dt):
                return timedelta(0)

        if self.name is 'aerosol':
            self.datatable['t_series_aerosol'] = (self.datatable['date'] + ' ' + self.datatable['time'])
            datetime_object = [datetime.strptime(str(date), '%d.%m.%Y %H:%M:%S') for date in
                               self.datatable['t_series_aerosol']]
            self.datatable.drop(['date'], axis=1, inplace=True)
            self.datatable.drop(['time'], axis=1, inplace=True)
            self.datatable.drop(['t_series_aerosol'], axis=1, inplace=True)

        elif self.name is 'wave_old':
            datetime_object = [datetime.strptime(str(date), '%d.%m.%Y %H:%M') for date in
                               self.datatable['t_series_waves']]
            self.datatable.drop(['t_series_waves'], axis=1, inplace=True)

        elif self.name is 'wave':
            datetime_object = [datetime.strptime(str(date), '%d-%b-%Y %H:%M:%S') for date in
                               self.datatable['t_series_waves']]
            self.datatable.drop(['t_series_waves'], axis=1, inplace=True)

        elif self.name is 'windspeed_metstation':
            datetime_object = [datetime.strptime(str(date), '%d.%m.%Y %H:%M') for date in
                               self.datatable['time_5min']]
            self.datatable.drop(['time_5min'], axis=1, inplace=True)

        elif self.name is 'windspeed_metstation_corrected':
            datetime_object = [datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S') for date in
                               self.datatable['timest_']]
            self.datatable.drop(['timest_'], axis=1, inplace=True)

        elif self.name is 'meteo_variable_track':
            datetime_object = [datetime.strptime(str(date), '%d.%m.%y %H:%M') for date in
                   self.datatable['timestamp']]
            self.datatable.drop(['timestamp'], axis=1, inplace=True)

        elif self.name is 'meteo_ecmwf_intpol':
            datetime_object = [datetime.strptime(str(date), '%d.%m.%y %H:%M') for date in
                   self.datatable['datetime']]
            self.datatable.drop(['datetime'], axis=1, inplace=True)

        elif self.name is 'sorpasso_variable_track':
            datetime_object = [datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S') for date in
                   self.datatable['date']]
            self.datatable.drop(['date'], axis=1, inplace=True)

        elif self.name is 'meteo_water_vapour':
            datetime_object = [datetime.strptime(str(date), '%d.%m.%y %H:%M') for date in
                   self.datatable['datetime']]
            self.datatable.drop(['datetime'], axis=1, inplace=True)


        datetime_obj = [date_.replace(tzinfo=UTC()) for date_ in datetime_object]
        timestamp = [timegm(date_.timetuple()) for date_ in datetime_obj]

        self.datatable[new_column_name] = pd.DataFrame(timestamp)

        del timestamp, datetime_obj

##############################################################################################################
    def set_datetime_index(self):
        self.convert_time_to_unixts('timest_')
        self.datatable['timest_'] = pd.to_datetime(self.datatable['timest_'], unit='s')
        self.datatable.set_index(pd.DatetimeIndex(self.datatable['timest_']), inplace=True)
        self.datatable.drop(['timest_'], axis=1, inplace=True)

##############################################################################################################
    def filter_particle_sizes(self, threshold=3, mode='full', NORM_METHOD='fancy', save=''):
        """ IN :
                threshold: check for increase in neighboring values (multiplicative)
                mode:   full: check within neighborhing bins AND neighborhing rows (time smoothness)
                        bin: check only withing neighboring bins
            OUT:
                returns filtered data table in the ACEdata object.

            EXAMPLE:
            partSize.filter_particlesizes(threhdold=5)
        """

        if self.name.lower() != 'particle_size_distribution':
            print('not the particle size data')
            return self

        # NORM_METHOD = 'fancy' # global_threshold, that should not be used if not for SPEEEEEEEDDDOAAAHH
        conds = []
        if NORM_METHOD.lower() == 'fancy':

            par_ = np.pad(self.datatable.values,(1,1), mode='symmetric')
            s1, s2 = par_.shape
            conds = np.zeros((s1-2,s2-2),dtype=bool)

            for ro in range(1,s1-1):

                rle = np.zeros((1,s2-2),dtype=bool)
                rri = np.zeros((1,s2-2),dtype=bool)
                rup = np.zeros((1,s2-2),dtype=bool)
                rdo = np.zeros((1,s2-2),dtype=bool)

                for co in range(1,s2-1):

                    if ~np.isnan((par_[ro,co-1],par_[ro,co])).any():#
                        rle[:,co-1] = (np.abs(np.log(1e-10 + par_[ro,co]) - np.log(1e-10 + par_[ro,co-1])) > np.log(threshold))

                    if ~np.isnan((par_[ro,co+1],par_[ro,co])).any():#(par_[ro,co],par_[ro,co+1])).any():
                        rri[:,co-1] = (np.abs(np.log(1e-10 + par_[ro,co]) - np.log(1e-10 + par_[ro,co+1])) > np.log(threshold))

                        if mode.lower == 'full':
                            if ~np.isnan((par_[ro-1,co],par_[ro,co])).any():#par_[ro,co],par_[ro-1,co])).any():
                                rup[:,co-1] = (np.abs(np.log(1e-10 + par_[ro,co]) - np.log(1e-10 + par_[ro-1,co])) > np.log(threshold))

                            if ~np.isnan((par_[ro+1,co],par_[ro,co])).any():#(par_[ro,co],par_[ro+1,co])).any():
                                rdo[:,co-1] = (np.abs(np.log(1e-10 + par_[ro,co]) - np.log(1e-10 + par_[ro+1,co])) > np.log(threshold))

                comb_ = [rle,rri,rup,rdo]
        #         print(comb_)
        #         print(np.any(comb_,axis=0))

                conds[ro-1,:] = np.any(comb_,axis=0)

            del comb_

        elif NORM_METHOD.lower() == 'global':
            conds = self.datatable > 10000
            conds = conds.values


        temp_ = self.datatable.copy()
        temp_.iloc[conds] = np.nan
        # bad_rows = temp_.loc[(temp_ > 8000).any(axis=1)].index
        bad_rows = np.where((temp_ > 300).any(axis=1))[0]
        conds[bad_rows,:] = True
        bad_rows = (temp_.isnull()).sum(axis=1) > 50
        print('nan bad rows (>50): ' + str(np.sum(bad_rows)))
        conds[bad_rows,:] = True

        self.particle_filtered = self.datatable.copy()
        self.particle_filtered.iloc[conds] = np.nan

        if save:
            print('saving in %s ...' %(save))
            self.particle_filtered.to_csv(save)

        del temp_, bad_rows
        # return self.particle_filtered

##############################################################################################################
def zeropad_date(x):
    return '0' + str(x) if len(x) < 2 else str(x)
##############################################################################################################
def parsetime_date(x):
    if len(x.split(' ')) > 1:
        x, b = x.split(' ')
        toadd = ' ' + b
    else:
        toadd = ''
    if x.count(':') == 2:
        h, m, s = x.split(':')
        h = zeropad_date(h)
        m = zeropad_date(m)
        s = zeropad_date(s)
        t_str =  h + ':' + m + ':' + s
    elif x.count(':') == 1:
        h, m = x.split(':')
        h = zeropad_date(h)
        m = zeropad_date(m)
        t_str = h + ':' + m + ':00'

    t_str += toadd
    return t_str
##############################################################################################################

def add_datetime_index_from_column(df, old_column_name, string_format='%d.%m.%Y %H:%M:%S', new_column_name='timest_'):

    """
        Convert something that looks like a data, as specified by string_format into a python datetime index
    """

    from datetime import datetime, tzinfo, timedelta
    #from time import mktime
    from calendar import timegm

    class UTC(tzinfo):
        """UTC subclass"""
        def utcoffset(self, dt):
            return timedelta(0)
        def tzname(self, dt):
            return "UTC"
        def dst(self, dt):
            return timedelta(0)

    datetime_object = [datetime.strptime(str(date), string_format) for date in df[old_column_name]]
    df.drop([old_column_name], axis=1, inplace=True)

    datetime_obj = [date_.replace(tzinfo=UTC()) for date_ in datetime_object]
    timestamp = [timegm(date_.timetuple()) for date_ in datetime_obj]

    df[new_column_name] = pd.DataFrame(timestamp)

    df['timest_'] = pd.to_datetime(df['timest_'], unit='s')
    df.set_index(pd.DatetimeIndex(df['timest_']), inplace=True)
    df.drop(['timest_'], axis=1, inplace=True)

    return df

##############################################################################################################
def add_legs_index(df, **kwargs):

    if 'leg_dates' not in kwargs:
        leg_dates = [['2016-12-20', '2017-01-21'], # leg 1
                    ['2017-01-22', '2017-02-25'],  # leg 2
                    ['2017-02-26', '2017-03-19']]  # leg 3
    else:
        leg_dates = kwargs['leg_dates']

    if 'codes' not in kwargs:
        codes = [1, 2, 3]
    else:
        codes = kwargs['codes']

    """Add a column to the datatable specifying the cruise leg"""
    assert len(codes) == len(leg_dates), "To each date interval must correspond only one code"

    if 'leg' in df:
        print('leg column already there')
        return df


    dd = pd.Series(data=np.zeros((len(df.index),)), index=df.index, name='leg')
    #df['leg'] = pd.Series()

    c = 0
    while c < len(codes):
        dd.loc[leg_dates[c][0]:leg_dates[c][1]] = codes[c]
        c += 1

    # dd.loc[dd['leg'].isnull()] = 0
    df = df.assign(leg=dd)
    return df

##############################################################################################################
def ts_aggregate_timebins(df1, time_bin, operations, mode='new'):
    """
        Outer merge of two datatables based on a common resampling of time intervals:
            INPUTS
                - df1      : DataFrames indexed by time
                - time_bin : Aggregation bin, in minutes
                - strategy : {'col': {'colname_min': np.min ...
                    dictionary of (possibly multiple) data aggregation strategies. New columns will have
                    corresponding subscript. df1 and df2 should be the input variable names
            OUTPUT
                - resampled dataframe with uninterrupted datetime index
            EXAMPLE

            operations = {'min': np.min , 'mean': np.mean, 'max': np.max,'sum': np.sum}
            df_res = dataset.ts_aggregate_timebins(df1, 15*60, operations)
            print(df_res.head())
            PREV VERSION LOGIC:
            df1_d = dict.fromkeys(df1.columns,[])
            for keys in df1_d:
                df1_d[keys]= {keys + op : operations[op] for op in operations}
    """
    res_ = str(time_bin) + 'T'


    if mode == 'old': # This one gets a FutureWarning! But the alternative is _very_ slow
        df1_d = dict.fromkeys(df1.columns,[])
        for keys in df1_d:
            df1_d[keys]= {keys + op : operations[op] for op in operations}

        out = df1.resample(res_).agg(df1_d)
        # loffeset = mean of timestamp window timedelta(minutes = time_bin/2)
        out.columns = out.columns.droplevel(level=0)

    else:
        def my_agg(x,ops):
            ops_cols = {}
            # print(x.columns.tolist())
            #ops_cols = {'hs_m': x['hs'].agg(np.mean),
            #            'hs_s': x['hs'].agg(np.std)}

            for cols in x.columns.tolist():
                for op, val in ops.items():
                    ops_cols[cols+op] = x[cols].agg(val)

            #print([key for key in ops_cols.keys()])
            # return ops_cols
            return pd.Series(ops_cols, index=[key for key in ops_cols.keys()])

        # print(my_agg(df1, operations))
        #out = df1.resample(res_).apply(my_agg)
        out = df1.groupby(pd.Grouper(freq=res_)).apply(my_agg,ops=operations)

    # out.columns = out.columns.droplevel(level=0)
    return out

##############################################################################################################
def filter_particle_sizes(datatable,threshold=3):

    par_ = np.pad(datatable.values,(1,1), mode='symmetric')
    s1, s2 = par_.shape
    conds = np.zeros((s1-2,s2-2),dtype=bool)

    for ro in range(1,s1-1):#1,s1-1):#s1):
        rle = np.zeros((1,s2-2),dtype=bool)
        rri = np.zeros((1,s2-2),dtype=bool)
        rup = np.zeros((1,s2-2),dtype=bool)
        rdo = np.zeros((1,s2-2),dtype=bool)
        for co in range(1,s2-1):

            if ~np.isnan((par_[ro,co],par_[ro,co-1])).any():
                rle[:,co-1] = (np.abs(par_[ro,co] - par_[ro,co-1])  > threshold)

            if ~np.isnan((par_[ro,co],par_[ro,co+1])).any():
                rri[:,co-1] = (np.abs(par_[ro,co] - par_[ro,co+1])  > threshold)

            if ~np.isnan((par_[ro,co],par_[ro-1,co])).any():
                rup[:,co-1] = (np.abs(par_[ro,co] - par_[ro-1,co])  > threshold)

            if ~np.isnan((par_[ro,co],par_[ro+1,co])).any():
                rdo[:,co-1] = (np.abs(par_[ro,co] - par_[ro+1,co])  > threshold)

        conds[ro-1,:] = rle# + rri + rup + rdo

    filtered[conds] = np.nan
    return filtered

##############################################################################################################
def generate_particle_data(data_folder='./data/', mode='all', data_output='./data/intermediate/',
                           savedata=False, filtering_parameter=3):
    '''
        Utility to get aerosol data
            INPUTS
                - data_folder: main directory where data is
                - data_output: where to store aggregated dataframe
                - mode: what to read and how:
                    - 'all' : all particle data in columns (single bins AND >400, >700 nm)
                    - 'single_bins' : only single bin data (without accumulated >400 and > 700nm)
                    - 'aggregated': all particle data aggregated in superbins (with >400,>700 nm)
                    - 'aggregated_no_noise' : as 'aggregated' but without noisy bins
                - filtering_parameter: used to define which filtered data to read (see filter_particle_size   threshold): 3, 5, 10 : the larger, the more permissive
                - savedata: store locally (in ./data/intermediate/) a copy of the assembled file
            OUTPUT
                - time-indexed DF with particle sizes
            EXAMPLE
    '''

    particles = pd.read_csv(data_folder + 'intermediate/7_aerosols/03_filtered_particle_size_distribution_T' + str(filtering_parameter) + '.csv')
    particles.set_index('timest_',inplace=True)
    particles.index = pd.to_datetime(particles.index,format='%Y-%m-%d %H:%M:%S')
    #print(particles.index)

    if mode.lower() == 'single_bins':
        if savedata:
            particles.to_csv(aggregated_no_noise + '/03_particles_' + mode + '.csv', sep=',', na_rep='')
        return particles

    aero_l700 = ACEdata(data_folder=data_folder + 'raw/7_aerosols/', name='aerosol')
    aero_l700 = aero_l700.datatable
    aero_l400 = pd.read_csv(data_folder + 'raw/7_aerosols/particles_greater_400nm.txt', na_values='NAN')
    aero_l400.index = aero_l700.index

    if mode.lower() == 'all':
        particles = particles.assign(newcol=aero_l400)
        particles = particles.rename(columns={'newcol': '>400'})
        particles = particles.assign(newcol=aero_l700)
        particles = particles.rename(columns={'newcol': '>700'})

        if savedata:
            particles.to_csv(data_output + '/particles_' + mode + '.csv', sep=',', na_rep='')
        return particles

    part_legend = pd.read_table(data_folder + 'raw/7_aerosols/04_diameterforfile_03.txt')

    if mode.lower() == 'aggregated':
        part_agg = pd.DataFrame()
        cond = (part_legend > 0) & (part_legend<=80)
        part_agg = part_agg.assign(newcol=particles.iloc[:,np.where(cond)[0]].sum(axis=1))
        part_agg = part_agg.rename(columns={'newcol': '11-80'})
        part_agg['11-80'].loc[np.sum(particles.iloc[:,np.where(cond)[0]].isnull(),axis=1) == cond.sum().values[0]] = np.nan

        cond = (part_legend>80) & (part_legend <= 200)
        part_agg = part_agg.assign(newcol=particles.iloc[:,np.where(cond)[0]].sum(axis=1))
        part_agg = part_agg.rename(columns={'newcol': '80-200'})
        part_agg['80-200'].loc[np.sum(particles.iloc[:,np.where(cond)[0]].isnull(),axis=1) == cond.sum().values[0]] = np.nan

        cond = (part_legend>200) & (part_legend <= 400)
        part_agg = part_agg.assign(newcol=particles.iloc[:,np.where(cond)[0]].sum(axis=1))
        part_agg = part_agg.rename(columns={'newcol': '200-400'})
        part_agg['200-400'].loc[np.sum(particles.iloc[:,np.where(cond)[0]].isnull(),axis=1) == cond.sum().values[0]] = np.nan

        part_agg['>400'] = aero_l400
        part_agg['>700'] = aero_l700

        if savedata:
            part_agg.to_csv(data_output + '/particles_' + mode + '.csv', sep=',', na_rep='')
        return part_agg

    if mode.lower() == 'aggregated_no_noise':

        part_agg = pd.DataFrame()

        cond = ((part_legend>0) & (part_legend<35)) | ((part_legend>51) & (part_legend<=80))
        part_agg = part_agg.assign(newcol=particles.iloc[:,np.where(cond)[0]].sum(axis=1))
        part_agg = part_agg.rename(columns={'newcol': '11-80'})
        part_agg['11-80'].loc[np.sum(particles.iloc[:,np.where(cond)[0]].isnull(),axis=1) == cond.sum().values[0]] = np.nan

        cond = (part_legend>80) & (part_legend <= 200)
        part_agg = part_agg.assign(newcol=particles.iloc[:,np.where(cond)[0]].sum(axis=1))
        part_agg = part_agg.rename(columns={'newcol': '80-200'})
        part_agg['80-200'].loc[np.sum(particles.iloc[:,np.where(cond)[0]].isnull(),axis=1) == cond.sum().values[0]] = np.nan

        cond = (part_legend>200) & (part_legend <= 300)
        part_agg = part_agg.assign(newcol=particles.iloc[:,np.where(cond)[0]].sum(axis=1))
        part_agg = part_agg.rename(columns={'newcol': '200-300'})
        part_agg['200-300'].loc[np.sum(particles.iloc[:,np.where(cond)[0]].isnull(),axis=1) == cond.sum().values[0]] = np.nan

        part_agg['>400'] = aero_l400
        part_agg['>700'] = aero_l700

        if savedata:
            part_agg.to_csv(data_output + '/particles_' + mode + '.csv', sep=',', na_rep='')
        return part_agg

##############################################################################################################
def read_standard_dataframe(data_folder, datetime_index_name='timest_'):

    data = pd.read_csv(data_folder)
    data.set_index('timest_', inplace=True)
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')
    #print(data.index)
    return data

##############################################################################################################
def subset_data_stack_variables(data, varset, seatype='total', mode='subset'):
    '''
        Subset variables of a dataset stack:
            INPUTS
                - data: dataframe containing data with way too many columns. if it contains the "parbin" column (used by baselinescripts to retrieve the label to refress on) is returned by default as last columns
                - varset: keyword specifiying the subset to get
                - seatype: defaults to total. When wave parameters are to be retrieved, specify which kind of sea parameters to haveself

            OUPUTS
                - dataframe with a subset of input columns
            EXAMPLE
                df = subset_data_stack_variables(stack, 'ecmwf')
    '''

    if (varset.lower() == 'wave')|(varset.lower() == 'waves'):
        cols_total = ['hs', 'tp', 'steep', 'phase_vel', 'age', 'wind']
        cols_wind  = ['hs_w', 'tp_w', 'steep_w', 'phase_vel_w', 'age_w', 'wind']
    elif varset.lower() == 'wave_nowind':
        cols_total = ['hs', 'tp', 'steep', 'phase_vel']
        cols_wind  = ['hs_w', 'tp_w', 'steep_w', 'phase_vel_w']
    elif varset.lower() == 'wave_reduced':
        cols_total = ['hs', 'tp', 'wind']
        cols_wind  = ['hs_w', 'tp_w', 'wind']
    elif varset.lower() == 'wave_ecmwf':
        cols_total = ['hs', 'tp', 'steep', 'phase_vel', 'age', 'wind',
         't', 'slp', 'q', 'v', 'u', 'iwc', 'ps', 'rh', 'th', 'lsp', 'cp',
         'rtot', 'blh', 'blhp', 'td2m', 't2m', 'skt', 'sif', 'the', 'lsm']
        cols_wind  = ['hs_w', 'tp_w', 'steep_w', 'phase_vel_w', 'age_w', 'wind', 't', 'slp', 'q', 'v', 'u', 'iwc', 'ps', 'rh', 'th', 'lsp', 'cp', 'rtot', 'blh', 'blhp', 'td2m', 't2m', 'skt', 'sif', 'the', 'lsm']
    elif varset.lower() == 'ecmwf':
        cols_total = ['t', 'slp', 'q', 'v', 'u', 'iwc', 'ps', 'rh', 'th', 'lsp', 'cp',
        'rtot', 'blh', 'blhp', 'td2m', 't2m', 'skt', 'sif', 'the', 'lsm']
    elif varset.lower() == 'wholeset':
        cols_total = data.columns.tolist()[:-1]
    else:
        print('variables not specified')
        return

    if seatype == 'total':
        cols = cols_total
    elif seatype == 'wind':
        cols = cols_wind
    else:
        print('seatype not correct')
        return

    if mode == 'subset':
        if 'parbin' in data.columns:
            cols.append('parbin')
        return data[cols]
    elif mode == 'returnnames':
        print(cols)
        return cols

##############################################################################################################
def retrieve_correlation_to_particles(data, particles, var, legs=[1,2,3], plots=True):
    '''
        Retrieve correlation to the whole series, and, if legs are specified, to independend legs.
    '''
    if 'leg' not in data.columns.tolist():
        data = add_legs_index(data)
    if 'leg' not in particles.columns.tolist():
        particles = add_legs_index(particles)

    corrs = {}
    N = []
    if legs:
        for leg in legs:
            for col in particles.columns.tolist()[:-1]:
                p_leg = particles.loc[particles.loc[:,'leg'] == leg,col]
                va_leg = data.loc[data.loc[:,'leg'] == leg, var]
                corrs['corr_' + col + '_' + str(leg)] = p_leg.corr(va_leg)
                # N[col + '_' + str(leg)] =
                N.append(np.sum(p_leg.notnull() & va_leg.notnull()))

    for col in particles.columns.tolist()[:-1]:

        p_leg = particles.loc[particles.loc[:,'leg'] != 0 ,col]
        va_leg = data.loc[data.loc[:,'leg'] != 0, var]
        corrs['corr_' + col + '_' + 'total'] = p_leg.corr(va_leg)
        #N[col + '_' + 'total'] =
        N.append(np.sum(p_leg.notnull() & va_leg.notnull()))

    if plots:
        if len(legs) != 3:
            print('still need to adapt to single series correlation, quick fix!')
            return corrs

        labels = [str(a) for a in corrs.keys()]
        labels = [s.split('_') for s in labels]

        bar_w = 0.16
        f,ax = plt.subplots(figsize=(7,4))
        ra = np.arange(4)
        n = np.arange(len(particles.columns.tolist())-1)

        legend = ['leg 1', 'leg 2', 'leg 3', 'whole series']
        in_ = 0
        for group in ra:
            end_ = in_ + len(ra) +1
            vals = list(corrs.values())[in_:end_]
            ax.bar(n+group*bar_w, vals, bar_w, label=legend[group])
            in_ = end_

        bars = [rect for rect in ax.get_children() if isinstance(rect, mpl.patches.Rectangle)]
        for b, bar in enumerate(bars):
            if b > len(N)-1:
                break

            height = bar.get_height()
            # print(b,N[b],N)
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2.0, height, N[b], ha='center', va='top')
            elif height < 0:
                plt.text(bar.get_x() + bar.get_width()/2.0, height, N[b], ha='center', va='bottom')


        ax.set_title('Correlations of ' + var + ' to particle groups')
        ax.set_xticks(n + 2*bar_w)
        ax.set_xticklabels(particles.columns.tolist())
        ax.set_ylim(-1,1)
        ax.set_ylabel('Correlation')
        ax.grid(which='major', axis='y', linestyle='--')
        ax.legend(loc=0)

    return corrs

##############################################################################################################
def read_traj_file_to_numpy(filename, ntime):
    '''
        traj_ensemble: Provides a datacube containing N_trajectories x M_backtracking_time_intervals x D_variables_model_out
        columns : name of D_variables_model_out
        starttime : beggining of the backtracking time series in the enseble
    '''
    with open(filename) as fname:
            header = fname.readline().split()
            fname.readline()
            variables = fname.readline().split()

    starttime = datetime.strptime(header[2], '%Y%m%d_%H%M')

    dtypes = ['f8']*(len(variables))
    dtypes[variables.index('time')] = 'datetime64[s]'

    convertfunc = lambda x: starttime + timedelta(**{'hours': float(x)})
    array = np.genfromtxt(filename, skip_header=5, names=variables,
                  missing_values='-999.99', dtype=dtypes,
                  converters={'time': convertfunc}, usemask=True)

    ntra = int(array.size / ntime)

    for var in variables:

        if (var == 'time'):
            continue
        array[var] = array[var].filled(fill_value=np.nan)
        # timestep, period = (array['time'][1] - array['time'][0], array['time'][-1] - array['time'][0])

    array = array.reshape((ntra, ntime))

    traj_ensemble = [line for nn in range (0,56) for it in range(0,81) for line in array[nn][it]]
    traj_ensemble = np.reshape(traj_ensemble,(ntra, ntime, -1))

    return traj_ensemble, variables, starttime
