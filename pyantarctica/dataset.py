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

##############################################################################################################
def parsetime(x):
    """
        Function to format timestamps to HH:MM:SS

        :param x: string containing badly formatted timestams
        :returns: nicely formatted timestamp string
    """

    def zeropad_date(x):
        """
            Helper to left-pad with 0s the timestamp

            :params x: timestamp string to be zeropadded, if len(x)<2
            :returns: left - 0 padded string of length 2
        """
        return '0' + str(x) if len(x) < 2 else str(x)

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
def add_datetime_index_from_column(df, old_column_name, string_format='%d.%m.%Y %H:%M:%S', index_name='timest_'):

    """
        Convert a column in a dataframe df to a datetime element and move it as index of the df. The string_format rule specifies how the raw string is formatted.

        :param df: dataframe of the original data
        :param old_column_name: column containing the raw datetime string to be converted
        :param string_format: how the string is organized
        :param index_name: new column to add with datetime, and moved as index
        :returns: dataframe with datetime index named index_name
    """

    from datetime import datetime, tzinfo, timedelta
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

    df[index_name] = pd.DataFrame(timestamp)
    df[index_name] = pd.to_datetime(df[index_name], unit='s')

    df.set_index(pd.DatetimeIndex(df[index_name]), inplace=True)
    df.drop([index_name], axis=1, inplace=True)

    return df

##############################################################################################################
def add_legs_index(df, **kwargs):

    """
        Add a leg identifier (defaults to [1,2,3]) to a datetime indexed dataframe of ACE measurements

        :param df: datetime indexed dataframe to which add the leg index
        :param kwargs: contains options to override defaults:

        |   leg_dates : array indicating beginning and end (columns) of time period belonging to a leg / subleg (number of rows equals number of legs / sublegs)
        |   codes : list indicating what code to use to denote legs. Defaults to integers in range [1,2,3,...] with 0 indicating out-of-leg datapoints

        :returns: dataframe with a new column "leg" indicating to which leg the datapoint belongs to

    """

    if 'leg_dates' not in kwargs:
        leg_dates = [['2016-12-20', '2017-01-21'], # leg 1
                    ['2017-01-22', '2017-02-25'],  # leg 2
                    ['2017-02-26', '2017-03-19']]  # leg 3
    else:
        leg_dates = kwargs['leg_dates']


    # merz_glacier = ['2017-01-29', '2017-01-31']

    if 'codes' not in kwargs:
        codes = np.arange(1,1+len(leg_dates))
    else:
        codes = kwargs['codes']

    """Add a column to the datatable specifying the cruise leg"""
    assert len(codes) == len(leg_dates), "To each date interval must correspond only one code"

    if 'leg' in df:
        print('leg column already there')
        return df

    dd = pd.Series(data=np.zeros((len(df.index))), index=df.index, name='leg')

    c = 0
    while c < len(codes):
        dd.loc[leg_dates[c][0]:leg_dates[c][1]] = codes[c]
        c += 1

    if 'merz_glacier' in kwargs:
        dd.loc[merz_glacier[0]:merz_glacier[1]] = -1

    # dd.loc[dd['leg'].isnull()] = 0
    df = df.assign(leg=dd)

    return df

##############################################################################################################
def ts_aggregate_timebins(df1, time_bin, operations, mode='new', index_position='middle'):
    """
        Outer merge of two datatables based on a common resampling of time intervals:

        :param df1: datetime indexed dataframe to resample
        :param time_bin: aggregation time, in minutes
        :param operations: dictionary containing a suffix to add to the column name (empty to keep same name in the returned df) and operation used to aggregate entries in the df. Example {'_min': np.nanmin, '_max':np.nanmax} returns a dataframe with columns '%colname_min' and '%colname_max', where %colname is the original column name.
        :param index_position: position of the new timestamp index wrt to the resampling : 'initial', 'middle' or 'final'
        :returns: resampled dataframe with uninterrupted datetime index and NaNs where needed
        :Example:

            operations = {'_min': np.min , '_mean': np.mean, '_max': np.max, '_sum': np.sum}
            df_res = dataset.ts_aggregate_timebins(df1, 15, operations, index_position='initial')
            print(df_res.head())
    """
    res_ = str(time_bin) + 'T'

    df = pd.DataFrame()

    for cols in df1.columns.tolist()[0:]:
        for op, val in operations.items():
            df[cols+op] = df1[cols].groupby(pd.Grouper(freq=res_)).agg(val)

            nanind = (df1[cols].fillna(1).groupby(pd.Grouper(freq=res_)).count() != df1[cols].groupby(pd.Grouper(freq=res_)).count()) & (df1[cols].groupby(pd.Grouper(freq=res_)).count() == 0)

            df[cols+op].iloc[np.where(nanind)] = np.nan

    if index_position == 'initial':
        time_shift = 0
    elif index_position == 'middle':
        time_shift = int(np.ceil(time_bin/2*60)) # From minutes to seconds, to round up to the second.
    elif index_position == 'final':
        time_shift = int(np.ceil(time_bin/2*60))

    # print(time_ shift)
    return df.shift(time_shift, 's')

##############################################################################################################
def filter_particle_sizes(pSize, threshold=3, window=3, mode='mean', save=''):
    """
        Function to filter particle size distribution dataframes, where rows are datetime index entries and columns are subsequent, as given by the instrument, particle size bins.

        :param pSize: datetime indexed dataframe of all particles. I think should be in dN/dlogD
        :param threshold: check for a multiplicative increase, given by expected_value * threshold
        :param window: size of the square moving windows to retrieve statistics, see ret_neigh()
        :param mode: string specifying how to compute value to threholds: 'mean': the expected value is the mean of the values in the window. 'std' returns the standard deviation and 'any' all the values (to implement original filtering method)
        :param save: if string is presented, save csv at this pointer
        :returns: filtered particle dataframe, with same datetime index as pSize
    """

    def ret_neigh(i,j, window = 3):
        """
            helper with different hardcoded scanning windows. returns a list of indexes based on the centerpoint of the window.

            .. todo:: automatize the window creation. This was a quick and dirt thing to see if it was working

        """

        if window == 1:
            fil_ = [[i, j], [i-1, j], [i+1,j]]

        if window == 2:
            fil_ = [[i, j],             [i-1, j],
                            [i,   j-1],           [i,  j+1],
                                        [i+1, j]]
        if window == 3:
            fil_ = [[i, j], [i-1, j-1], [i-1, j], [i-1,j+1],
                            [i,   j-1],           [i,  j+1],
                            [i+1, j-1], [i+1, j], [i+1,j+1]]
        elif window == 5:
            fil_ = [[i, j], [i-2, j-2], [i-2, j-1], [i-2,j], [i-2, j+1], [i-2, j+2],
                            [i-1, j-2], [i-1, j-1], [i-1,j], [i-1, j+1], [i-1, j+2],
                            [i,   j-2], [i,   j-1],          [i,   j+1], [i,   j+2],
                            [i+1, j-2], [i+1, j-1], [i+1,j], [i+1, j+1], [i+1, j+2],
                            [i+2, j-2], [i+2, j-1], [i+2,j], [i+2, j+1], [i+2, j+2]]

        return fil_


    def ret_values(data, window=3):
        """
            helper returning all the values in the data matrix contained in every window, as rows. it returns a matrix with size data.shape[0]*data.shape[1], and number of columns eequal to the number of elements in the scanning windows. Note that the center valie is returned at the first row, row=0.
            .. todo:: make the inner loop faster by using actual index sampling, the clever way
        """
        em = np.zeros((data.shape[0]*data.shape[1],window**2))
        d_pad = np.pad(data,(1,1), mode='symmetric')
        col = 0
        for i in np.arange(1,d_pad.shape[0]-1):
            for j in np.arange(1,d_pad.shape[1]-1):
                fil_ = ret_neigh(i,j,window)
                em[col][:] = [d_pad[fil_[c][0]][fil_[c][1]] for c in range(window**2)]
                col += 1
        return em

    part_size = pSize.copy().values
    vect_filt = ret_values(part_size)

    # print(mea)
    if mode=='mean':
        mea = np.nanmean(vect_filt[:,1:],axis=1)
        local_t  = vect_filt[:,0] > (threshold * mea)
    elif mode=='std':
        mea = np.nanmean(vect_filt[:,1:],axis=1)
        sig = np.nanstd(vect_filt[:,1:])
        local_t = vect_filt[:,0] > (mea + threshold * sig)
        local_t = local_t|(vect_filt[:,0] < (mea - threshold * sig))
    elif mode=='any':
        local_t = [np.any(vect_filt[jj,0] > threshold * vect_filt[jj,1:]) for jj in range(vect_filt.shape[0])]
        local_t = np.array(local_t)

    loca = np.reshape(local_t, part_size.shape)
    part_size[loca] = np.nan

    # Delete lines which are alone (e.g. nans above and nans below)
    global_t = np.zeros((part_size.shape[0],1), dtype=bool)
    for i in range(1,part_size.shape[0]-1):
        line_above_na = np.sum(np.isnan(part_size[i-1,:])) > part_size.shape[1]*0.50
        line_above_ze = np.sum(part_size[i-1,:] == 0) > part_size.shape[1]*0.50
        line_above = line_above_na | line_above_ze

        line_below_na = np.sum(np.isnan(part_size[i+1,:])) > part_size.shape[1]*0.50
        line_below_ze = np.sum(part_size[i+1,:] == 0) > part_size.shape[1]*0.50
        line_below = line_below_na | line_below_ze

        global_t[i] = (line_above & line_below)
        # global_t[i] = global_t[i] | (np.sum(np.isnan(part_size[i,:])) > part_size.shape[1]*0.50)
        global_t[i] = global_t[i] | np.any(part_size[i,:] > 500)

    part_size[np.where(global_t)[0],:] = np.nan

    string_ = 'local_t : ' + str(np.sum(local_t)) + '; ' + 'global_t : ' + str(np.sum(global_t)) + '; '

    print(string_)

    df = pd.DataFrame(part_size,index=pSize.index,columns=pSize.columns)

    if save:
        df.to_csv(save)

    return df

##############################################################################################################
def generate_particle_data(data_folder='../data/', mode='all', data_output='./data/intermediate/',
                           savedata=False, filtering_parameter=3):
    '''
        Utility to get aggregated aerosol data and store it in a df where columns are the different accregation. This function recomputes directly from locally stored filtered particle size data the aggregations. This function assumes that the folder structures are fixed.


        :param data_folder: pointer to data top directory
        :param data_output: where to save aggregated dataframe
        :param mode: what to read and how:

        | 'all' : all particle data in columns (small particles single bins and >400, >700 nm)
        | 'single_bins' : only single bin data (without accumulated >400 and > 700nm)
        | 'aggregated': all particle data aggregated in superbins (with >400,>700 nm)
        | 'aggregated_no_noise' : as 'aggregated' but without noisy bins

        :param filtering_parameter: used to define which filtered data to read (see filter_particle_size   threshold): 3, 5, 10 : the larger, the more permissive
        :param savedata: boolean, store locally (in ./data/intermediate/) a copy of the assembled file
        :returns: datetime-indexed dataframe with particle sizes as columns

        .. todo:: remove data folder structure assumption. Or actually rethink the whole function.
        .. todo:: actually maybe better to deprecate the whole function and prepare a notebook preparing the data directly, maybe easier for renku inclusion. Even better, a big script preparing the different datsets.

    '''

    particles = pd.read_csv(data_folder + 'intermediate/7_aerosols/03_filtered_particle_size_distribution_T' + str(filtering_parameter) + '.csv')
    particles.set_index('timest_',inplace=True)
    particles.index = pd.to_datetime(particles.index,format='%Y-%m-%d %H:%M:%S')
    #print(particles.index)

    if mode.lower() == 'single_bins':
        if savedata:
            particles.to_csv(aggregated_no_noise + '/03_particles_' + mode + '.csv', sep=',', na_rep='')
        return particles

    aero_l700 = read_standard_dataframe(data_folder + 'intermediate/7_aerosols/12_Aerosol_larger_700nm_postprocessed.csv')
    aero_l400 = read_standard_dataframe(data_folder +  'intermediate/7_aerosols/13_Aerosol_larger_400nm_postprocessed.csv')

    if mode.lower() == 'all':
        particles = particles.assign(newcol=aero_l400)
        particles = particles.rename(columns={'newcol': '>400'})
        particles = particles.assign(newcol=aero_l700)
        particles = particles.rename(columns={'newcol': '>700'})

        if savedata:
            particles.to_csv(data_output + '/00_particles_' + mode + '.csv', sep=',', na_rep='')
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
        part_agg.to_csv(data_output + '/00_particles_' + mode + '.csv', sep=',', na_rep='')

    return part_agg

##############################################################################################################
def read_standard_dataframe(data_folder, datetime_index_name='timest_', crop_legs=True):
    '''
        Helper function to read a ``*_postprocessed.csv`` file, and automatically crop out leg 1 - leg 3, and set as datetime index a specific column (or defaults to the standard)

        :param data_folder: from where to read the ``*_postprocessed.csv`` datafile
        :param datetime_index_name: specify non-default datetime index column (= ``timest_``)
        :param crop_legs: boolean to specify whether to remove data outside leg 1 to leg 3
        :returns: dataframe containing the original data, leg-cropped (if option active)
    '''
    data = pd.read_csv(data_folder, na_values=' ')
    data.set_index(datetime_index_name, inplace=True)
    data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')

    if crop_legs:
        data = add_legs_index(data)
        data = data[data['leg'] != 0]
        data = data.drop('leg',axis=1)

    return data

##############################################################################################################
def subset_data_stack_variables(data, varset, seatype='total', mode='subset'):
    '''
        Subset variables of a dataset stack. This function is just a shorthand to not have to specify all vaiables when subsetting dataframes. Not the most clever way to deal with this honestly.

        :param data: dataframe of data to subset
        :param varset: keyword specifiying the subset to get
        :param seatype: defaults to total. When "wave" parameters are to be retrieved, specify which kind of sea parameters to have
        :param mode: subset : return actual dataframe or returnnames: return only column names per subset
        :returns: either dataframe of data, or list of columns names.
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

        :param data: dataframe of data, from which a column will be picked (var) and used to compute correlations to particle data
        :param particles: dataframe containing different particle sizes bins, to which compute correlation with data
        :param var: string indicating name of the data column to use
        :param legs: IDs of the legs in which you want to computer correlations
        :param plots: If True, plot the barplots indicating amount of correlations
        :returns: dictionary containing correlations to particles, figure (no handles returned, get them via plt.gca, plt.gcf and so on)
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
        Read air mass trajectory files as provided in project 11 -- read them from trajetories/lsl folder

        :param filename: file to be parsed
        :param ntime: numer of backtracking time points (i.e. how many rows per trajectory to read)
        :returns: traj_ensemble, a datacube containing N_trajectories x M_backtracking_time_intervals x D_variables_model_out
        :returns: columns : name of D_variables_model_out (variables provided by the lagranto model)
        :returns: starttime : beggining of the backtracking time series in the ensemble
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
