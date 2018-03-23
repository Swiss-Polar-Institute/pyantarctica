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
    
        # Lines are NOT 0-indexed
        if self.name is 'wave_old':
            self.dataname = 'waveData_Toffoli_all' 
            extension = '.txt'
            column_head = 23
            body = 24
            nantype = 'nan'
            ncols = 12
            delimiter = '\t'
            self.fullfolder = self.data_folder + self.raw_folder + '/' + self.dataname + extension 

        elif self.name is 'wave':
            self.dataname = 'wavedata-14.03-fromMat'
            extension = '.csv'
            column_head = 1
            body = 2
            nantype = 'NaN'
            ncols = 13
            delimiter = ','
            self.fullfolder = self.data_folder + self.intermediate_folder + '/' + self.dataname + extension 
            
        elif self.name is 'aerosol':
            self.dataname = 'AerosolData_Schmale_all'
            extension = '.txt'
            column_head = 11 
            body = 12#146
            nantype = ''
            delimiter = '\s+'
            self.fullfolder = self.data_folder + self.raw_folder + '/' + self.dataname + extension 
        
        elif self.name is 'windspeed_metstation':
            self.dataname = 'windspeed_metstation'
            extension = '.csv'
            column_head = 5 
            body = 6
            nantype = ''
            delimiter = ';'
            self.fullfolder = self.data_folder + self.raw_folder + '/' + self.dataname + extension 
         
        elif self.name is 'windspeed_metstation_corrected':
            self.dataname = 'windspeed_metstation_corrected'
            extension = '.csv'
            column_head = 1
            body = 2
            nantype = ''
            delimiter = ','
            self.fullfolder = self.data_folder + self.intermediate_folder + '/' + self.dataname + extension 
            
        else:
            print('dataset not handled yet: must be ''wave'' or ''aerosol''')

        self.datatable = self.load(column_head, body, delimiter, nantype)

    def load(self, column_head, body, delim, nantype):
        """Load and sets data object named 'dataset' """
        # fullfolder = self.data_folder + self.raw_folder + '/' + self.dataname + ext
        datatable = pd.read_table(self.fullfolder, skip_blank_lines=False, header=column_head-1,
                                skiprows=range(column_head,body-1), na_values=nantype, 
                                delimiter=delim, index_col=False)
        
        datatable.columns = [x.lower() for x in datatable.columns]

        if self.name is 'aerosol':
            print('fixing columns...')
            inds = datatable['time'].str.contains('NaN', na=True)
            datatable.loc[inds, 'time'] = datatable.loc[inds, 'date']
            datatable.loc[inds, 'date'] = datatable.loc[inds, 'num_conc']
            datatable.loc[inds, 'num_conc'] = np.nan
            datatable['num_conc'] = pd.to_numeric(datatable['num_conc'])
            
        elif self.name is 'wave':
            datatable.rename(columns={"date": "t_series_waves"}, inplace=True)
        
        elif self.name is 'windspeed_metstation':
            datatable.rename(columns={"windspeed (m/s)": "wind_metsta"}, inplace=True)

        print('Data successfully loaded from ' + self.fullfolder)
        
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
                               self.datatable['timestamp']]
            self.datatable.drop(['timestamp'], axis=1, inplace=True)
            

        datetime_obj = [date_.replace(tzinfo=UTC()) for date_ in datetime_object]
        timestamp = [timegm(date_.timetuple()) for date_ in datetime_obj]

        self.datatable[new_column_name] = pd.DataFrame(timestamp)
    
    def set_datetime_index(self): 
        self.convert_time_to_unixts('timest_')
        self.datatable['timest_'] = pd.to_datetime(self.datatable['timest_'], unit='s')
        self.datatable.set_index(pd.DatetimeIndex(self.datatable['timest_']), inplace=True)
        self.datatable.drop(['timest_'], axis=1, inplace=True)

# -------------------------

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
        return
    
    df['leg'] = pd.Series()
        
    c = 0
    while c < len(codes):
        df.loc[leg_dates[c][0]:leg_dates[c][1], 'leg'] = codes[c]
        c += 1
        
    df.loc[df['leg'].isnull(), 'leg'] = 0
    return df
        
def ts_aggregate_timebins(df1, time_bin, operations):
    """
        Outer merge of two datatables based on a common resampling of time intervals: 
            INPUTS
                - df1      : DataFrames indexed by time
                - time_bin : Aggregation bin, in seconds
                - strategy : {'col': {'colname_min': np.min ...
                    dictionary of (possibly multiple) data aggregation strategies. New columns will have
                    corresponding subscript. df1 and df2 should be the input variable names
            OUTPUT
                - resampled dataframe with uninterupted datetime index
            EXAMPLE 
                
            operations = {'min': np.min , 'mean': np.mean, 'max': np.max,'sum': np.sum}
            df_res = dataset.ts_aggregate_timebins(df1, 15*60, operations)
            print(df_res.head())
    """
    
    res_ = str(time_bin) + 'S'
    
    df1_d = dict.fromkeys(df1.columns,[])
    for keys in df1_d:
        df1_d[keys]= {keys + op : operations[op] for op in operations}
    
    out = df1.resample(res_).agg(df1_d)
    out.columns = out.columns.droplevel(level=0)
    return out

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



