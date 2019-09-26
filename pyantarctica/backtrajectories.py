# collection of functions to handle back trajectory data
import os
from pathlib import Path
from collections import defaultdict

import datetime 


import numpy as np
import pandas as pd


def BT_subsetting(Dir_L1, BT_file_name, Max_hours, COLUMN_SELECTION):
    df_BT = pd.read_csv(os.path.join(Dir_L1,BT_file_name),skiprows=[0,1,3,4],sep=r'\s+',na_values=['-999.99','-999.990'], engine='python') #SL added NaN detection 
    # there was some issue with spacing of the raw data file and its converstion to CSV fomrat with (probably) sep=r'\s*
    # sl: changed to r'\s+' to get rid of warning
    df_BT_loc=df_BT[df_BT['time']==0]
    min_inx=df_BT_loc.index.values
    if max(np.diff(np.diff(min_inx)))>0:
        print('warning for '+BT_file_name+'inconsitent BT length!')
    max_inx=min_inx+int(np.mean(np.diff(min_inx))) 
    
    df_BT['U10'] = np.sqrt(np.square(df_BT['U10'])+np.square(df_BT['V10'])) #
    df_BT.drop(columns=['V10'], inplace=True)


    BT_data_list = [] # container for selected BTs
    
    # select only BT which start within the current BLH
    # BTcutoff
    if 0:
        jBT_start_within_BLH = list(np.where(df_BT['p'][min_inx]>(df_BT['BLHP'][min_inx])[0]) )
        BT = df_BT.iloc[ df_BT.index<max_inx[jBT_start_within_BLH[-1]] ]   # subselect only BT starting within the BLH
    else:
        jBT_start_within_BLH = list(np.where(df_BT['p'][min_inx+1]>(df_BT['BLHP'][min_inx+1])) )[0]
        # require that first BT value is still within BL, this also rejects the cases where the BT crashes rigth away.
        if len(jBT_start_within_BLH)==0:
            # no bt passes, relax to p[0]>BLHP[0]
            jBT_start_within_BLH = list(np.where(df_BT['p'][min_inx]>(df_BT['BLHP'][min_inx])) )[0]
        #BT = df_BT.iloc[ df_BT.index<max_inx[jBT_start_within_BLH[-1]] ]   # subselect only BT starting within the BLH
        BT = df_BT.iloc[ ((df_BT.index<max_inx[jBT_start_within_BLH[-1]]) & (df_BT.index>=min_inx[jBT_start_within_BLH[0]]) )  ]   # subselect only BT starting within the BLH

    # select only selcted columns
    if COLUMN_SELECTION[0]=='all':
        BT = BT
    else:
        BT = BT[ COLUMN_SELECTION ]   

    # drop hours passe Max_hours 
    BT=BT[BT['time']>=-Max_hours]

    # add timest_ column with UTC time stamp of BT data row
    datestr0 = BT_file_name.find('201') # kee on the decade
    Date = BT_file_name[datestr0:datestr0+8]
    hour=np.float(BT_file_name[datestr0+9:datestr0+12])
    date1 = datetime.datetime.strptime(Date, "%Y%m%d")+datetime.timedelta(hours=hour)
            
    #if BT_file_name[0:7]=='lsl_u10':
    #    Date = BT_file_name.split("_")[2::3]
    #    hour=np.array(BT_file_name.split("_")[3::4]).astype(np.float)
    #else:
    #    Date = BT_file_name.split("_")[1::2]
    #    hour=np.array(BT_file_name.split("_")[2::3]).astype(np.float)
    #date1 = datetime.datetime.strptime(Date[0], "%Y%m%d")+datetime.timedelta(hours=hour[0])
    BT_date_time = np.array([date1 + datetime.timedelta(hours=BT['time'].iloc[i]) for i in range(len(BT))])

    # add BTindex to distinquish the backtrajectories
    BTindex = np.ones_like(BT['p']) # add back trajectory index
    for jBT in jBT_start_within_BLH:
        BTindex[(BT.index>=min_inx[jBT]) & (BT.index<max_inx[jBT])] = jBT
    
    BT['timest_'] = BT_date_time
    BT['BTindex'] = BTindex
    
    # BT = new dataframe combining all selected data from the selected back trajectories
    return BT



def bt_time_till_cond(bt,cond,label='time_till_cond', return_full_info=False):
    # bt average of the time over which the condition is true
    # output data frame containse one column with label=label provided
    # if return_full_info==True the output will be a mulit columns data frame
    # time_till_cond_count 	time_till_cond 	time_till_cond_std 	time_till_cond_min 	time_till_cond_25% 	time_till_cond_50% 	time_till_cond_75% 	time_till_cond_max
    
    bt_LSM = bt[['timest_', 'time', 'BTindex']].copy()
    bt_LSM[label]=cond
    bt_LSM[label]=bt_LSM.groupby(['timest_', 'BTindex']).aggregate('cumprod')[label] # cumprod to find till when the condition is true
    bt_LSM[label]=bt_LSM[label]*bt_LSM['time']
    if return_full_info:
        bt_LSM = bt_LSM.groupby(['timest_', 'BTindex']).aggregate('max').groupby('timest_').describe()
        bt_LSM.drop(columns=['time'], inplace=True) # remove unneeded colums
        bt_LSM.columns = ['_'.join(col).strip() for col in bt_LSM.columns.values]
        bt_LSM.rename(columns={label+'_mean': label}, inplace=True) # rename to drop the _mean (for consitency)
        bt_LSM.rename(columns={label+'_count': 'bt_count'}, inplace=True) # rename to bt_count (for consitency)
        x=list((bt_LSM.columns[1:].values))
        x.insert(1,bt_LSM.columns[0])
        bt_LSM = bt_LSM.reindex(columns=x)
    else:
        bt_LSM = bt_LSM.groupby(['timest_', 'BTindex']).aggregate('max').groupby('timest_').mean()#[value]
        bt_LSM.drop(columns=['time'], inplace=True) # remove unneeded colums

    return bt_LSM


def bt_count_bt(bt):
    bt_ = bt[['timest_', 'BTindex']][bt['time']==0.0].copy()
    bt_['bt_count']=1
    bt_ = bt_.groupby(['timest_']).sum()#['bt_count']
    #bt_['bt_count']=bt_.groupby(['timest_']).sum()['bt_count']
    #bt_.set_index(pd.DatetimeIndex(bt_.timest_), inplace=True )
    bt_.drop(columns='BTindex',inplace=True)
    return bt_




def bt_accumulate_weights(bt,w,W_str='W'):
    W_depo = bt[['timest_', 'BTindex','time']].copy()
    W_depo[W_str] = w.values
    W_depo[W_str]=W_depo.groupby(['timest_', 'BTindex']).aggregate('cumprod')[W_str]
    if 1:
        W_depo[W_str]=W_depo[W_str]/w.values # remove first weight
        W_depo[W_str][w.values==0.0]=0.0
    return W_depo

def bt_get_values(bt_, return_value, return_times, aggr_trajs, aggr_time):
    # aggregates data over trajectories
    # option to calculated '', 'cumsum', or 'cumprod' along time axis
    # returns array of data shape (len(return_times), len(bt_.timest_) )
    timest_=pd.DatetimeIndex(np.unique(bt_.timest_))
    bt_ = bt_.groupby(['timest_', 'time']).aggregate(aggr_trajs) # average over the trajectories
    
    # build a wrapper for percentiles: (runs a few seconds!)
    #bt_ = bt_.groupby(['timest_', 'time']).aggregate(percentile(50))
    
    if len(aggr_time)>0:
        bt_=bt_.groupby('timest_').aggregate(aggr_time) # 

    if aggr_time in ['', 'cumsum', 'cumprod']:
        x_=[]
        for return_time in return_times:
            x_.append(bt_.iloc[bt_.index.get_level_values('time') == np.float(return_time)][return_value].values)

        x_ = pd.DataFrame( np.transpose(x_), index=timest_)

    else:
        x_ = pd.DataFrame( bt_[return_value].values, columns=[return_value], index=pd.DatetimeIndex(np.unique(bt.timest_)) )
        x.index.rename('timest_', inplace=True)
    return x_
    

def bt_plot_values(bt_, return_value, return_times, aggr_trajs, aggr_time, Nhours=1):

    x_ = bt_get_values(bt_, return_value, return_times, aggr_trajs, aggr_time)

    if Nhours>1:
        x_ = x_.resample(str(int(Nhours))+'H').mean()
    
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(12, 4))
    ax=axes
    plt.pcolormesh(x_.index, return_times/24,   np.transpose(x_.values) )
    cb=plt.colorbar(label=return_value)

    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_minor_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    ax.fmt_xdata = DateFormatter('%Y-%m-%d')
    fig.autofmt_xdate()
    
    ax.set_ylabel('air mass age [d]')
    
    
    leg_dates = [['2016-12-20', '2017-01-21'], # leg 1
                    ['2017-01-22', '2017-02-25'],  # leg 2
                    ['2017-02-26', '2017-03-19']]  # leg 3
    ax.set_xlim([pd.to_datetime(leg_dates[0][0]), pd.to_datetime(leg_dates[-1][1])])
    for leg_date in np.reshape(leg_dates,(6,1)):
        ax.axvline(x=pd.to_datetime(leg_date), color='gray', linestyle='--')
    
    return fig, ax, cb