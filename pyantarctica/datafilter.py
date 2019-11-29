import numpy as np
import pandas as pd

from datetime import datetime, timedelta

def outliers_iqr_score(ys):
    """
        Function to identification of outliers based on the IQR.

        :param ys: arry of data
        :returns: score=series of deviation from the iqr, same shape like ys
        :returns: iqr=the iqr of ys

    """
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5) # was 1.5
    upper_bound = quartile_3 + (iqr * 1.5) # was 1.5

    score3 = np.max( [(ys-quartile_3)/iqr, 0*ys] , axis=0 )
    score1 = np.max( [(quartile_1-ys)/iqr, 0*ys] , axis=0 )
    score = np.max( [score1, score3] , axis=0 )

    if iqr==0:
        score = np.abs(ys-np.median(ys))

    return score, iqr

def outliers_iqr_time_window(X,Nmin=60,minN=12,d_phys=0, d_iqr=1.5):
    """
        Function to identification of outliers from a time series based on the IQR.
        Data are considered outlier if they deviate from the local median by more than d_iqr*IQR and at least d_phys (allow for white noise)

        :param X: dataframe with one data field
        :param Nmin: expected time of physical variations in minutes
        :param minN: minimum number of samples required within the time window specified by Nmin, in order to filter
        :param d_phys: expected white noise on the data
        :param d_iqr: maximum allowed deviation from the median in units of iqr

        :returns: score=series of deviation from the iqr, same shape like ys
        :returns: outlier.values, = boolean vector flaging the outliers with True
        :returns: np.squeeze(X_iqr.values),
        :returns: np.squeeze(X_delta.values)

    """
    # use iqr outlier detection, for iqr=0 use {y-median(y)}>d_phys to identyfy outliers
    # uses outliers_iqr_score

    # NEED TO ADD TZ-naivisation of X!
    X_iqr = X.copy()*np.NaN
    X_delta = X.copy()*np.NaN
    X_=X.resample(str(Nmin)+'T', loffset = timedelta(seconds=Nmin*30), base=0).mean() # for nearest/first no offset required
    binL = X_.index[0]-timedelta(seconds=Nmin*30)
    binU = X_.index[-1]+timedelta(seconds=Nmin*30)
    Dbin = timedelta(seconds=Nmin*60)
    t = X.index+timedelta(seconds=0)

    for V_str in X.columns:
        V = X[V_str]
        QC=~np.isnan(V)
        for nbin1 in np.arange(binL,binU,Dbin):
            binL_current = pd.to_datetime(nbin1)#.tz_localize('UTC')
            in_bin_t = (t>binL_current) & (t<=binL_current+Dbin)
            in_bin=in_bin_t & QC
            if sum(in_bin)>minN:
                delta, iqr = outliers_iqr_score(V[in_bin==1])
                X_iqr[V_str][in_bin==1] = iqr
                X_delta[V_str][in_bin==1] = delta

    # evaluate
    if d_phys>0:
        outlier = ( (X_iqr[V_str]*X_delta[V_str])>d_phys) & (X_delta[V_str]>d_iqr ) | ( (X_delta[V_str]>d_phys ) & (X_iqr[V_str]==0) )
    else:
        outlier = (X_delta[V_str]>d_iqr )
    if 1:
        # repeat with 1/2 shifted windows to avoid edges to be detected as outliers
        binL = X_.index[0]
        binU = X_.index[-1]+timedelta(seconds=Nmin*60)
        Dbin = timedelta(seconds=Nmin*60)

        for V_str in X.columns:
            V = X[V_str]
            QC=~np.isnan(V)
            for nbin1 in np.arange(binL,binU,Dbin):
                #in_bin_t = (t>pd.to_datetime(nbin1)) & (t<=pd.to_datetime(nbin1)+Dbin)
                binL_current = pd.to_datetime(nbin1)#.tz_localize('UTC')
                in_bin_t = (t>binL_current) & (t<=binL_current+Dbin)
                in_bin=in_bin_t & QC
                if sum(in_bin)>minN:
                    delta, iqr = outliers_iqr_score(V[in_bin==1])
                    X_iqr[V_str][in_bin==1] = iqr
                    X_delta[V_str][in_bin==1] = delta

        # evaluate
        if d_phys>0:
            outlier2 = ( (X_iqr[V_str]*X_delta[V_str])>d_phys) & (X_delta[V_str]>d_iqr )  | ( (X_delta[V_str]>d_phys ) & (X_iqr[V_str]==0) )
        else:
            outlier2 = (X_delta[V_str]>d_iqr )

        outlier = outlier & outlier2

    return outlier.values, np.squeeze(X_iqr.values), np.squeeze(X_delta.values)


def APS_filter_single_counts(APS):
    # function to remove the intervals with the lowest APS counts
    # At large size bins we reach the APS digital resolution with lot of measurements showing the same low values
    # this function should be applied AFTER the APS has been resampled to the desired time resolution
    #
    #APS[(APS == 0).sum(axis=1)==len(APS.columns.values)]=np.nan # some rows are all zero -> set them to nan, appears no longer relevant
    #
    APS_RESOLUTION = np.nanmin(APS.values) # get the resolution of the APS by looking at the smallest count
    if (APS_RESOLUTION-0.003855343373493976)>0.001:
        print('waring the lowest APS count is not the one expected!!!!')
    # remove all counts smaller than 3x resolution
    APS_RESOLUTION=0.003855343373493976
    for APS_str in APS.columns:
        APS.at[APS[APS_str]<3*APS_RESOLUTION, APS_str]=np.NaN

    return APS
