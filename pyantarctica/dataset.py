#
# Copyright 2017-2018 - Swiss Data Science Center (SDSC) and ACE-DATA/ASAID Project consortium.
# A partnership between École Polytechnique Fédérale de Lausanne (EPFL) and
# Eidgenössische Technische Hochschule Zürich (ETHZ). Written within the scope
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
import pickle

# import matplotlib as mpl
# import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from pathlib import Path

##############################################################################################################
def parsetime(x):
    """
        Function to format timestamps to HH:MM:SS

        :param x: string containing badly formatted timestams
        :returns: nicely formatted timestamp string
    """

    if len(x.split(" ")) > 1:
        x, b = x.split(" ")
        toadd = " " + b
    else:
        toadd = ""
    if x.count(":") == 2:
        h, m, s = x.split(":")
        h = zeropad_date(h)
        m = zeropad_date(m)
        s = zeropad_date(s)
        t_str = h + ":" + m + ":" + s
    elif x.count(":") == 1:
        h, m = x.split(":")
        h = zeropad_date(h)
        m = zeropad_date(m)
        t_str = h + ":" + m + ":00"

    t_str += toadd

    return t_str


def zeropad_date(x):
    """
        Helper to left-pad with 0s the timestamp

        :params x: timestamp string to be zeropadded, if len(x)<2
        :returns: left - 0 padded string of length 2
    """
    return "0" + str(x) if len(x) < 2 else str(x)


##############################################################################################################
def add_datetime_index_from_column(
    df, old_column_name, string_format="%d.%m.%Y %H:%M:%S", index_name="timest_"
):

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

    datetime_object = [
        datetime.strptime(str(date), string_format) for date in df[old_column_name]
    ]
    df.drop([old_column_name], axis=1, inplace=True)

    datetime_obj = [date_.replace(tzinfo=UTC()) for date_ in datetime_object]
    timestamp = [timegm(date_.timetuple()) for date_ in datetime_obj]

    df[index_name] = pd.DataFrame(timestamp)
    df[index_name] = pd.to_datetime(df[index_name], unit="s")

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

    if "leg_dates" not in kwargs:
        # leg_dates = [['2016-12-20', '2017-01-21'], # leg 1
        #             ['2017-01-22', '2017-02-25'],  # leg 2
        #             ['2017-02-26', '2017-03-19']]  # leg 3
        # updates leg dates with hourly resolution
        leg_dates = [
            ["2016-11-19 23:00", "2016-12-15 05:00"],  # leg 0
            ["2016-12-20 13:00", "2017-01-18 22:00"],  # leg 1
            ["2017-01-22 13:00", "2017-02-22 09:00"],  # leg 2
            [
                "2017-02-26 02:00",
                "2017-03-19 07:00",
            ],  # leg 3, ship at full speed @'2017-02-26 02:00', in the vicinity at '2017-03-18 14:00'
            ["2017-03-22 20:00", "2017-04-11 13:00"],  # leg 4
        ]

    else:
        leg_dates = kwargs["leg_dates"]

    # merz_glacier = ['2017-01-29', '2017-01-31']

    if "codes" not in kwargs:
        codes = np.arange(1, 1 + len(leg_dates))
        codes = [0, 1, 2, 3, 4]  # SL
    else:
        codes = kwargs["codes"]

    """Add a column to the datatable specifying the cruise leg"""
    assert len(codes) == len(
        leg_dates
    ), "To each date interval must correspond only one code"

    if "leg" in df:
        # SL:#print("leg column already there")
        # SL:#return df
        df.drop(
            columns=["leg"], inplace=True
        )  # SL:# Changed here to ALWAYS OVERWRITE the "leg" field

    dd = pd.Series(data=np.zeros((len(df.index))) * np.nan, index=df.index, name="leg")

    c = 0
    while c < len(codes):
        dd.loc[leg_dates[c][0] : leg_dates[c][1]] = codes[c]
        c += 1

    if "merz_glacier" in kwargs:
        dd.loc[merz_glacier[0] : merz_glacier[1]] = -1

    #  dd.loc[dd['leg'].isnull()] = 0
    df = df.assign(leg=dd)

    return df


##############################################################################################################
def ts_aggregate_timebins(
    df1, time_bin, operations, mode="new", index_position="middle"
):
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
    if type(time_bin) == int:
        res_ = str(time_bin) + "T"
    elif type(time_bin) == str:
        res_ = time_bin + "T"
    else:
        print("time bin in wrong type!")
        return None

    df = pd.DataFrame()

    for cols in df1.columns.tolist()[0:]:
        for op, val in operations.items():
            df[cols + op] = df1[cols].groupby(pd.Grouper(freq=res_)).agg(val)

            nanind = (
                df1[cols].fillna(1).groupby(pd.Grouper(freq=res_)).count()
                != df1[cols].groupby(pd.Grouper(freq=res_)).count()
            ) & (df1[cols].groupby(pd.Grouper(freq=res_)).count() == 0)

            df[cols + op].iloc[np.where(nanind)] = np.nan

    if index_position == "initial":
        time_shift = 0
    elif index_position == "middle":
        time_shift = int(
            np.ceil(time_bin / 2 * 60)
        )  # From minutes to seconds, to round up to the second.
    elif index_position == "final":
        time_shift = int(np.ceil(time_bin / 2 * 60))

    # print(time_ shift)
    return df.shift(time_shift, "s")


##############################################################################################################
def read_standard_dataframe(
    data_folder,
    datetime_index_name="timest_",
    crop_legs=True,
    date_time_format="%Y-%m-%d %H:%M:%S",
):
    """
        Helper function to read a ``*_postprocessed.csv`` file, and automatically crop out leg 1 - leg 3, and set as datetime index a specific column (or defaults to the standard)

        :param data_folder: from where to read the ``*_postprocessed.csv`` datafile
        :param datetime_index_name: specify non-default datetime index column (= ``timest_``)
        :param crop_legs: boolean to specify whether to remove data outside leg 1 to leg 3
        :param date_time_format: string to sepcify the date_time_format (added 2020.04.23 by SL)
        :returns: dataframe containing the original data, leg-cropped (if option active)
    """
    data = pd.read_csv(data_folder, na_values=" ", sep=",")
    data.set_index(datetime_index_name, inplace=True)
    data.index = pd.to_datetime(data.index, format=date_time_format)

    if crop_legs:
        leg = add_legs_index(data)["leg"]
        crop_out_ind = np.zeros(leg.shape)
        for ll in range(3):
            crop_out_ind += leg == ll + 1
        data = data[crop_out_ind.astype(bool)]  #  1-3
        # data = data.drop('leg',axis=1)
    return data


##############################################################################################################
def read_traj_file_to_numpy(filename, ntime):
    """
        Read air mass trajectory files as provided in project 11 -- read them from trajetories/lsl folder

        :param filename: file to be parsed
        :param ntime: numer of backtracking time points (i.e. how many rows per trajectory to read)
        :returns: traj_ensemble, a datacube containing N_trajectories x M_backtracking_time_intervals x D_variables_model_out
        :returns: columns : name of D_variables_model_out (variables provided by the lagranto model)
        :returns: starttime : beggining of the backtracking time series in the ensemble
    """
    with open(filename) as fname:
        header = fname.readline().split()
        fname.readline()
        variables = fname.readline().split()

    starttime = datetime.strptime(header[2], "%Y%m%d_%H%M")

    dtypes = ["f8"] * (len(variables))
    dtypes[variables.index("time")] = "datetime64[s]"

    convertfunc = lambda x: starttime + timedelta(**{"hours": float(x)})
    array = np.genfromtxt(
        filename,
        skip_header=5,
        names=variables,
        missing_values="-999.99",
        dtype=dtypes,
        converters={"time": convertfunc},
        usemask=True,
    )

    ntra = int(array.size / ntime)

    for var in variables:

        if var == "time":
            continue
        array[var] = array[var].filled(fill_value=np.nan)
        # timestep, period = (array['time'][1] - array['time'][0], array['time'][-1] - array['time'][0])

    array = array.reshape((ntra, ntime))

    traj_ensemble = [
        line for nn in range(0, 56) for it in range(0, 81) for line in array[nn][it]
    ]
    traj_ensemble = np.reshape(traj_ensemble, (ntra, ntime, -1))

    return traj_ensemble, variables, starttime


from datetime import datetime, timedelta

##############################################################################################################
def match2series(ts, ts2match):
    """
        Function to crop/append to the series ts in order to have the same number of samples as ts2match
        REQUIRES the two indices to be same for ts2match and ts !!!
        TODO add a warning if this is not the case!!!
        this is just a wrapper around pandas merge (import pandas as pd)

        :param ts: datetime indexed dataframe to be modified
        :param ts2match: datetime indexed dataframe, of which the index shall be taken

        :Example:

            wind = match2series(wind,aerosols) # output is the wind matched to aerosols
    """
    ts2match = ts2match[ts2match.columns[0]].to_frame()
    ts2match.rename(index=str, columns={ts2match.columns[0]: "var2match"}, inplace=True)
    ts = pd.merge(ts, ts2match, left_index=True, right_index=True, how="right")
    ts = ts.drop(columns=["var2match"])
    return ts


##############################################################################################################


def resample_timeseries(
    ts,
    time_bin,
    how="mean",
    new_label_pos="c",
    new_label_parity="even",
    old_label_pos="c",
    old_resolution=0,
    COMMENTS=False,
):
    """
        Function to resample timeseries to multiple of minutes placing the inter val label where you like it
        Info on time series location in the inital ts can be used to ensure accurate binning
        Output time stamp label position left, right, centre and parity ('even' -> index=00:05:00, 'odd' -> index=00:02:30) can be choosen

        :param ts: datetime indexed dataframe to resample
        :param time_bin: integer aggregation time, in minutes
        :param how: string specifyin how to aggregate. Has to be compatible with df.resample('5T').aggregate(how)
        :param old_label_pos: string ('l'=left, 'r'=right, 'c'=center) position of the initial timestamp
        :param old_resolution: integer input time resolution in minutes used to correct input time stamp if (old_label_pos=='c')==False, set to 0 if unknown
        :param new_label_pos: string ('l'=left, 'r'=right, 'c'=center), define if timest_ label denotes left, right, center of new intervals
        :param new_label_parity: string ('even' -> index=00:05:00, 'odd' -> index=00:02:30), if time stamp will look like as when resample would be run ('even')
        :param COMMENTS: boolean (True->print some info about what is done)
        :returns: resampled dataframe with uninterrupted datetime index and NaNs where needed
        :Example:

            ts_mean = dataset.resample_timeseries(ts, 15, how='mean')
            # if you know that initial ts lable was on right of interval and resolution was 5min (e.g. aerosols)
            ts_mean = dataset.resample_timeseries(ts, 15, how='mean', old_label_pos='r', old_resolution=5)

    """

    # assume input time series has label on interval center, then
    # we can resample to new time series with label position in center of interval,
    # choose if you like the lable to look like 'even' -> index=00:05:00, 'odd' -> index=00:02:30
    if (new_label_pos == "c") & (new_label_parity == "even"):
        # put label on center, index=00:05:00
        ts_offset = timedelta(minutes=(time_bin / 2))
        rs_loffset = timedelta(minutes=0)
    elif (new_label_pos == "c") & (new_label_parity == "odd"):
        # put label on center, index=00:02:30
        ts_offset = timedelta(minutes=0)
        rs_loffset = timedelta(minutes=(time_bin / 2))
    elif (new_label_pos == "l") & (new_label_parity == "even"):
        # put label on left, index=00:05:00 (classic resample behaviour)
        ts_offset = timedelta(minutes=0)
        rs_loffset = timedelta(minutes=0)
    elif (new_label_pos == "l") & (new_label_parity == "odd"):
        # put label on left, index=00:02:30
        ts_offset = timedelta(minutes=-(time_bin / 2))
        rs_loffset = timedelta(minutes=+(time_bin / 2))
    elif (new_label_pos == "r") & (new_label_parity == "even"):
        # put label on right end of new resample interval, index=00:05:00
        ts_offset = timedelta(minutes=+time_bin)
        rs_loffset = timedelta(minutes=0)
    elif (new_label_pos == "r") & (new_label_parity == "odd"):
        # put label on right end of new resample interval, index=00:02:30
        ts_offset = timedelta(minutes=0)
        rs_loffset = timedelta(minutes=+time_bin)
    else:
        print('new_label_pos must be either "l","r", or "c"!')
        print('new_label_parity must be either "odd" or "even"!')
        return

    # now check if the old lable pos is not 'c' and add an offset to ts_offset to correct for this

    if (old_label_pos == "c") == False:
        if old_resolution > 0:
            # known old_resolution we can use it to calcualte the offset to add
            if old_label_pos == "l":
                ts_offset = ts_offset + timedelta(minutes=+(old_resolution / 2))
            elif old_label_pos == "r":
                ts_offset = ts_offset + timedelta(minutes=-(old_resolution / 2))
        else:
            tres_ = np.median(np.diff(ts.index.tolist())).total_seconds()
            tres_ = int(tres_)  # round to full second
            if COMMENTS == True:
                print(
                    "Inferring old time resolution to be "
                    + str(tres_)
                    + " seconds ("
                    + str(tres_ / 60)
                    + " minutes)"
                )
            if old_label_pos == "l":
                ts_offset = ts_offset + timedelta(seconds=+(tres_ / 2))
            elif old_label_pos == "r":
                ts_offset = ts_offset + timedelta(seconds=-(tres_ / 2))
            else:
                print('old_label_pos must be either "l","r", or "c"')
                return

    # fix the initial index if it is needed,
    ts_resampled = ts.copy()
    ts_resampled.index = ts_resampled.index + ts_offset
    # resample with desired offset (the loffset changes the lable after resample has acted on the time series)
    ts_resampled = ts_resampled.resample(
        str(time_bin) + "T", loffset=rs_loffset
    ).aggregate(how)
    return ts_resampled


def get_raw_param(
    VarNameLUT="u10", META_FILE="../data/ASAID_DATA_OVERVIEW - Sheet1.csv"
):

    """
        Function to read the not resampled time series of one parameter based on META_FILE

        :param VarNameLUT: string defining the Variable name (VarNameLUT column)
        :param META_FILE: File containing pointers to files
        :returns: dataframe containing the time series
    """
    META_FILE = Path(META_FILE)
    META = pd.read_csv(META_FILE, sep=",")

    Proj_folder = META["Proj"][META["VarNameLUT"] == VarNameLUT].values[0]
    FilenameIntermediate = (
        META["FilenameIntermediate"][META["VarNameLUT"] == VarNameLUT].values[0]
        + "_parsed.csv"
    )
    VarNameIntermediate = META["VarNameIntermediate"][
        META["VarNameLUT"] == VarNameLUT
    ].values[0]
    Resolution = META["Resolution"][META["VarNameLUT"] == VarNameLUT].values[0]
    timest_loc = META["timest_loc"][META["VarNameLUT"] == VarNameLUT].values[0]

    if FilenameIntermediate in [
        "01_waves_recomputed_parsed.csv"
    ]:  # catch files not ending on parsed
        FilenameIntermediate = "01_waves_recomputed.csv"
        var = read_standard_dataframe(
            Path("..", "data", "intermediate", Proj_folder, FilenameIntermediate)
        )[[VarNameIntermediate]]
    elif FilenameIntermediate in [
        "iDirac_Isoprene_MR_All_Legs_parsed.csv"
    ]:  # catch files not ending on parsed
        FilenameIntermediate = "iDirac_Isoprene_MR_All_Legs.csv"
        var = read_standard_dataframe(
            Path("..", "data", "intermediate", Proj_folder, FilenameIntermediate)
        )[[VarNameIntermediate]]
    elif FilenameIntermediate in ["02_hplc_pigments_parsed.csv"]:
        var = read_standard_dataframe(
            Path("..", "data", "intermediate", Proj_folder, FilenameIntermediate)
        )
        var = var[var["Depth_m"] < 10]  # only use data from shallow depth <10meter
        var = var[[VarNameIntermediate]].sort_index()
    else:
        var = read_standard_dataframe(
            Path("..", "data", "intermediate", Proj_folder, FilenameIntermediate)
        )[[VarNameIntermediate]]
    var.rename(columns={VarNameIntermediate: VarNameLUT}, inplace=True)

    var.sort_index(inplace=True)
    return var


def filter_parameters(
    time_bin=60,
    LV_param_set_Index=1,
    LV_params=["u10"],
    META_FILE="../data/ASAID_DATA_OVERVIEW - Sheet1.csv",
    INTERPOLATE_limit=0,
    FILTER_LOD_OUTLIERS=True,
):
    """
        Function to read paramters for one LV experiment based on META_FILE
        All parameters are resampled to a common time stamp

        :param time_bin: integer, resampling time in minutes
        :LV_param_set_Index: integer, defining the index of the LV parameter set 1->LatentVar1, IF LV_param_set_Index==-1, LV_params is used instead
        :param LV_params: list of strings for manual defining the Variable names (VarNameLUT column), IGNORED if LV_param_set_Index~=-1
        :META_FILE: path to the META info file with columns
            'Proj', 'VarNameIntermediate', 'VarNameLUT', 'FilenameRaw', 'FilenameIntermediate',
            'Resolution', 'timest_loc', 'Samples', 'Unit', 'Description',
            'LatentVar0', 'LatentVar1', 'LatentVar2', 'LatentVar3'
        :returns: dataframe containing the time series
    """
    META_FILE = Path(META_FILE)

    META = pd.read_csv(META_FILE, sep=",")
    if LV_param_set_Index == -1:
        LV_params = LV_params  # use input parameter list
    elif LV_param_set_Index == "all":
        LV_params = list(META["VarNameLUT"].dropna().values)
    else:
        # define parameter list from ASAID_DATA_OVERVIEW.csv
        LV_params = list(
            META["VarNameLUT"][
                META["LatentVar" + str(LV_param_set_Index)] == 1.0
            ].values
        )

    params = []
    for VarNameLUT in LV_params:  #
        # print(VarNameLUT)
        Proj_folder = META["Proj"][META["VarNameLUT"] == VarNameLUT].values[0]
        FilenameIntermediate = (
            META["FilenameIntermediate"][META["VarNameLUT"] == VarNameLUT].values[0]
            + "_parsed.csv"
        )
        VarNameIntermediate = META["VarNameIntermediate"][
            META["VarNameLUT"] == VarNameLUT
        ].values[0]
        Resolution = META["Resolution"][META["VarNameLUT"] == VarNameLUT].values[0]
        timest_loc = META["timest_loc"][META["VarNameLUT"] == VarNameLUT].values[0]

        #### DISTRIBUTION TRANSFORMATION SERIES
        ditype = META["Distribution type"][META["VarNameLUT"] == VarNameLUT].values[0]

        # if FilenameIntermediate in ['01_waves_recomputed_parsed.csv']: # catch files not ending on parsed
        #    FilenameIntermediate = '01_waves_recomputed.csv'
        #    var = read_standard_dataframe(Path('..','data','intermediate',Proj_folder,FilenameIntermediate))[[VarNameIntermediate]]
        # elif FilenameIntermediate in ['iDirac_Isoprene_MR_All_Legs_parsed.csv']: # catch files not ending on parsed
        #    FilenameIntermediate = 'iDirac_Isoprene_MR_All_Legs.csv'
        #    var = read_standard_dataframe(Path('..','data','intermediate',Proj_folder,FilenameIntermediate))[[VarNameIntermediate]]
        # elif FilenameIntermediate in ['02_hplc_pigments_parsed.csv']:
        #    var = read_standard_dataframe(Path('..','data','intermediate',Proj_folder,FilenameIntermediate))
        #    var=var[var['Depth_m']<10] # only use data from shallow depth <10meter
        #    var=var[[VarNameIntermediate]].sort_index()
        # else:
        #    var = read_standard_dataframe(Path('..','data','intermediate',Proj_folder,FilenameIntermediate))[[VarNameIntermediate]]
        #
        # use get_raw_param here, instead of dublicating code. It is a bit overkill of releoding META_FILE each time. Could change to handing META over to get_raw_param instead of the file name
        var = get_raw_param(VarNameLUT=VarNameLUT, META_FILE=META_FILE)

        if VarNameIntermediate in ["seaice"]:
            # for sea ice interpolate only values, where the interpolation results to 0 values
            # we don't interpolate accross the strech of missing data where the ship was parked at the Mertz glacier
            var.at[
                (
                    np.isnan(var[var.columns[0]])
                    & (var[var.columns[0]].interpolate(direction="both") == 0)
                ),
                var.columns[0],
            ] = 0

        if VarNameIntermediate in ["CL1", "CL2", "CL3"]:
            var = var.replace({np.inf: 1.5 * np.nanmax(var.replace({np.inf: 0}))})
        # Add alert for other infinity values

        #### - - - - - - - - - - - - - -  - ###
        # var += 1
        # if np.any(var < 0):
        #     print(f"{VarNameLUT}: {np.mean(var[var<0])}")
        #     var = var + np.nanmin(var)

        #### - FILTERING OF OUTLIERS / BELOW LOD VALUES ####
        LOD_SCALE = 1  # 1/np.sqrt(2) # suggested by Andrea Baccarini
        # TODO : use np.random to generate random numbers e.g. between 0.5 and 1 to be multiplied with the LOD to fill the samples
        if FILTER_LOD_OUTLIERS:
            LOD = META["LOD applied"][META["VarNameLUT"] == VarNameLUT].values[0]
            if ~np.isnan(LOD):
                var[var < LOD] = LOD * LOD_SCALE

        if ditype == "Log-Normal":
            var[var == 0] = np.nan
            # if np.any(var < 0):
            #     print(f"{VarNameLUT}: {np.mean(var[var<0])}")
            #     var = var + np.nanmin(var)

            var = np.log(var)

        #### - - - - - - - - - - - - - -  - ###

        # print(len(var), VarNameLUT, ditype)

        #### - TIME SERIES RESAMPLING TO GET DESIRED UNIFORM TEMPORAL RESOLUTION ####
        if VarNameIntermediate in ["CL1", "CL2", "CL3"]:
            # NEED TO DECIDE WHAT TO DO WITH CL!!
            # var.at[var[VarNameIntermediate]==np.Inf, VarNameIntermediate] = 10000# set a high value for infinite cloud level
            var = resample_timeseries(
                var,
                time_bin=time_bin,
                how="median",
                new_label_pos="c",
                new_label_parity="even",
                old_label_pos=timest_loc,
                old_resolution=Resolution,
                COMMENTS=False,
            )
        elif VarNameIntermediate in ["longitude"]:
            # special treatment of "longitude" to deal with the crossing of the dateline
            var_x = resample_timeseries(
                np.cos(var / 180 * np.pi),
                time_bin=time_bin,
                how="mean",
                new_label_pos="c",
                new_label_parity="even",
                old_label_pos=timest_loc,
                old_resolution=Resolution,
                COMMENTS=False,
            )
            var_y = resample_timeseries(
                np.sin(var / 180 * np.pi),
                time_bin=time_bin,
                how="mean",
                new_label_pos="c",
                new_label_parity="even",
                old_label_pos=timest_loc,
                old_resolution=Resolution,
                COMMENTS=False,
            )
            var = np.arctan2(var_y, var_x) * 180 / np.pi
        else:
            var = resample_timeseries(
                var,
                time_bin=time_bin,
                how="mean",
                new_label_pos="c",
                new_label_parity="even",
                old_label_pos=timest_loc,
                old_resolution=Resolution,
                COMMENTS=False,
            )
        var.rename(columns={VarNameIntermediate: VarNameLUT}, inplace=True)
        # #### - - - - - - - - - - - - - -  - ###

        # #### - FILTERING OF OUTLIERS / BELOW LOD VALUES ####
        # LOD_SCALE = 1  # 1/np.sqrt(2) # suggested by Andrea Baccarini
        # # TODO : use np.random to generate random numbers e.g. between 0.5 and 1 to be multiplied with the LOD to fill the samples
        # if FILTER_LOD_OUTLIERS:
        #     LOD = META["LOD applied"][META["VarNameLUT"] == VarNameLUT].values[0]
        #     if ~np.isnan(LOD):
        #         var[var < LOD] = LOD * LOD_SCALE

        # #### - - - - - - - - - - - - - -  - ###

        #### - add optional interpolation - ###

        if VarNameIntermediate in ["seaice"]:
            # for sea ice interpolate only values, where the interpolation results to 0 values
            # we don't interpolate accross the strech of missing data where the ship was parked at the Mertz glacier
            var.at[
                (
                    np.isnan(var[var.columns[0]])
                    & (var[var.columns[0]].interpolate(direction="both") == 0)
                ),
                var.columns[0],
            ] = 0

        if INTERPOLATE_limit > 0:

            if FilenameIntermediate in [
                "output-HV_analysis(fice)_v20190926_parsed.csv",
                "output-LV_analysis(fice)_v20190926_parsed.csv",
            ]:
                interp_limit = (
                    int(np.floor(Resolution / time_bin / 2)) + INTERPOLATE_limit
                )  # limit interpolation to 1/2 of the averaging window on each side
                var.interpolate(
                    limit=interp_limit,
                    limit_direction="both",
                    method="nearest",
                    inplace=True,
                )
            elif int(np.floor((Resolution / time_bin) / 2)) > 0:
                # if old resolution is below new resolution interpolate near the resampled values
                interp_limit = (
                    int(np.floor((Resolution / time_bin) / 2)) + INTERPOLATE_limit
                )  # limit interpolation to 1/2 of the averaging window on each side
                var.interpolate(
                    limit=interp_limit,
                    limit_direction="both",
                    method="linear",
                    inplace=True,
                )
            else:
                # if old resolution is same or higher than new resolution interpolate INTERPOLATE_limit steps to the side
                interp_limit = INTERPOLATE_limit  #
                var.interpolate(
                    limit=interp_limit,
                    limit_direction="both",
                    method="linear",
                    inplace=True,
                )
        #### - - - - - - - - - - - - - -  - ###

        # add the variable to the parameter frame
        if len(params) == 0:
            params = var
        else:
            params = pd.merge(
                params, var, left_index=True, right_index=True, how="outer"
            )

    return params

