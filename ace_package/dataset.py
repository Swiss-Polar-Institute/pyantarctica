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
            self.fullfolder = self.data_folder + self.raw_folder + '/' + self.dataname + extension 
        
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

        else:
            print('dataset not handled yet.')

            
        self.datatable = self.load(column_head, body, delimiter, nantype, col_name)

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
                                names=nam) # 
        
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
         
        # print(self.datatable.columns.tolist())
        # print(self.datatable.head())

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
        del timestamp, datetime_obj
        
    def set_datetime_index(self): 
        self.convert_time_to_unixts('timest_')
        self.datatable['timest_'] = pd.to_datetime(self.datatable['timest_'], unit='s')
        self.datatable.set_index(pd.DatetimeIndex(self.datatable['timest_']), inplace=True)
        self.datatable.drop(['timest_'], axis=1, inplace=True)
        
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
        return df
    
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
                - resampled dataframe with uninterrupted datetime index
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

    #     print(rle + rri + rup + rdo)
    #     print(rle)
    #     print(rri)
    #     print(rup) 
    #     print(rdo)

    #     rup = [np.abs(par_[ro,co] - par_[ro-1,co])  > 3 for co in range(1,s2-1) if (par_[ro,co],par_[ro-1,co]) is not np.nan]
    #     rri = [np.abs(par_[ro,co] - par_[ro,co+1])  > 3 for co in range(1,s2-1) if (par_[ro,co],par_[ro,co+1]) is not np.nan]
    #     rup = [np.abs(par_[ro,co] - par_[ro-1,co])  > 3 for co in range(1,s2-1) if (par_[ro,co],par_[ro-1,co]) is not np.nan]
    #     rdo = [np.abs(par_[ro,co] - par_[ro+1,co])  > 3 for co in range(1,s2-1) if (par_[ro,co],par_[ro+1,co]) is not np.nan]

    #         rle.append(np.abs(par_[ro,co] - par_[ro,co-1])  > 3)
    #         rri.append(np.abs(par_[ro,co] - par_[ro,co+1])  > 3)
    #         rup.append(np.abs(par_[ro,co] - par_[ro-1,co])  > 3)
    #         rdo.append(np.abs(par_[ro,co] - par_[ro+1,co])  > 3)

