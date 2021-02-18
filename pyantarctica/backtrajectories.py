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

import os

# from pathlib import Path
# from collections import defaultdict

import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# import matplotlib.colors as colors

# import matplotlib.path as mpath  # for round plot
from matplotlib.dates import (
    MonthLocator,
    DayLocator,
    HourLocator,
    DateFormatter,
    # drange,
)


def BT_subsetting(Dir_L1, BT_file_name, Max_hours, COLUMN_SELECTION):
    df_BT = pd.read_csv(
        os.path.join(Dir_L1, BT_file_name),
        skiprows=[0, 1, 3, 4],
        sep=r"\s+",
        na_values=["-999.99", "-999.990"],
        engine="python",
    )  # SL added NaN detection
    # there was some issue with spacing of the raw data file and its converstion to CSV fomrat with (probably) sep=r'\s*
    # sl: changed to r'\s+' to get rid of warning
    df_BT_loc = df_BT[df_BT["time"] == 0]
    min_inx = df_BT_loc.index.values
    if max(np.diff(np.diff(min_inx))) > 0:
        print("warning for " + BT_file_name + "inconsitent BT length!")
    max_inx = min_inx + int(np.mean(np.diff(min_inx)))

    df_BT["U10"] = np.sqrt(np.square(df_BT["U10"]) + np.square(df_BT["V10"]))  #
    df_BT.drop(columns=["V10"], inplace=True)

    BT_data_list = []  # container for selected BTs

    # select only BT which start within the current BLH
    # BTcutoff
    if 0:
        jBT_start_within_BLH = list(
            np.where(df_BT["p"][min_inx] > (df_BT["BLHP"][min_inx])[0])
        )
        BT = df_BT.iloc[
            df_BT.index < max_inx[jBT_start_within_BLH[-1]]
        ]  # subselect only BT starting within the BLH
    else:
        jBT_start_within_BLH = list(
            np.where(df_BT["p"][min_inx + 1] > (df_BT["BLHP"][min_inx + 1]))
        )[0]
        # require that first BT value is still within BL, this also rejects the cases where the BT crashes rigth away.
        if len(jBT_start_within_BLH) == 0:
            # no bt passes, relax to p[0]>BLHP[0]
            jBT_start_within_BLH = list(
                np.where(df_BT["p"][min_inx] > (df_BT["BLHP"][min_inx]))
            )[0]
        # BT = df_BT.iloc[ df_BT.index<max_inx[jBT_start_within_BLH[-1]] ]   # subselect only BT starting within the BLH
        BT = df_BT.iloc[
            (
                (df_BT.index < max_inx[jBT_start_within_BLH[-1]])
                & (df_BT.index >= min_inx[jBT_start_within_BLH[0]])
            )
        ]  # subselect only BT starting within the BLH

    # select only selcted columns
    if COLUMN_SELECTION[0] == "all":
        BT = BT
    else:
        BT = BT[COLUMN_SELECTION]

    # drop hours passe Max_hours
    BT = BT[BT["time"] >= -Max_hours]

    # add timest_ column with UTC time stamp of BT data row
    datestr0 = BT_file_name.find("201")  # kee on the decade
    Date = BT_file_name[datestr0 : datestr0 + 8]
    hour = np.float(BT_file_name[datestr0 + 9 : datestr0 + 12])
    date1 = datetime.datetime.strptime(Date, "%Y%m%d") + datetime.timedelta(hours=hour)

    # if BT_file_name[0:7]=='lsl_u10':
    #    Date = BT_file_name.split("_")[2::3]
    #    hour=np.array(BT_file_name.split("_")[3::4]).astype(np.float)
    # else:
    #    Date = BT_file_name.split("_")[1::2]
    #    hour=np.array(BT_file_name.split("_")[2::3]).astype(np.float)
    # date1 = datetime.datetime.strptime(Date[0], "%Y%m%d")+datetime.timedelta(hours=hour[0])
    BT_date_time = np.array(
        [date1 + datetime.timedelta(hours=BT["time"].iloc[i]) for i in range(len(BT))]
    )

    # add BTindex to distinquish the backtrajectories
    BTindex = np.ones_like(BT["p"])  # add back trajectory index
    for jBT in jBT_start_within_BLH:
        BTindex[(BT.index >= min_inx[jBT]) & (BT.index < max_inx[jBT])] = jBT

    BT["timest_"] = BT_date_time
    BT["BTindex"] = BTindex

    # BT = new dataframe combining all selected data from the selected back trajectories
    return BT


def bt_time_till_cond(bt, cond, label="time_till_cond", return_full_info=False):
    # bt average of the time over which the condition is true
    # output data frame containse one column with label=label provided
    # if return_full_info==True the output will be a mulit columns data frame
    # time_till_cond_count 	time_till_cond 	time_till_cond_std 	time_till_cond_min 	time_till_cond_25% 	time_till_cond_50% 	time_till_cond_75% 	time_till_cond_max

    bt_LSM = bt[["timest_", "time", "BTindex"]].copy()
    bt_LSM[label] = cond
    bt_LSM[label] = bt_LSM.groupby(["timest_", "BTindex"]).aggregate("cumprod")[
        label
    ]  # cumprod to find till when the condition is true
    bt_LSM[label] = bt_LSM[label] * bt_LSM["time"]
    if return_full_info:
        bt_LSM = (
            bt_LSM.groupby(["timest_", "BTindex"])
            .aggregate("max")
            .groupby("timest_")
            .describe()
        )
        bt_LSM.drop(columns=["time"], inplace=True)  # remove unneeded colums
        bt_LSM.columns = ["_".join(col).strip() for col in bt_LSM.columns.values]
        bt_LSM.rename(
            columns={label + "_mean": label}, inplace=True
        )  # rename to drop the _mean (for consitency)
        bt_LSM.rename(
            columns={label + "_count": "bt_count"}, inplace=True
        )  # rename to bt_count (for consitency)
        x = list((bt_LSM.columns[1:].values))
        x.insert(1, bt_LSM.columns[0])
        bt_LSM = bt_LSM.reindex(columns=x)
    else:
        bt_LSM = (
            bt_LSM.groupby(["timest_", "BTindex"])
            .aggregate("max")
            .groupby("timest_")
            .mean()
        )  # [value]
        bt_LSM.drop(columns=["time"], inplace=True)  # remove unneeded colums

    return bt_LSM


def bt_count_bt(bt):
    bt_ = bt[["timest_", "BTindex"]][bt["time"] == 0.0].copy()
    bt_["bt_count"] = 1
    bt_ = bt_.groupby(["timest_"]).sum()  # ['bt_count']
    # bt_['bt_count']=bt_.groupby(['timest_']).sum()['bt_count']
    # bt_.set_index(pd.DatetimeIndex(bt_.timest_), inplace=True )
    bt_.drop(columns="BTindex", inplace=True)
    return bt_


# signal loss calculations
def w_inside_mbl(BT):
    Tol = 1.05  # tolerate the pressure to be 10% lower than BLHP
    w = (
        (BT["p"] * Tol > BT["BLHP"])
    ) * 1  # weight to set N->0 if trajectory from outside BLH(p<BLHP)
    # toto: add toleration of single 3h exiting of BLH
    return w


def w_dry_deposition(BT, Dp):
    # caculate deposition weith for 1 diameter Dp and return as Dataframe
    # note in this implementation w_dry depends mostly on BLH!
    # (more loss for smaler BLH)
    # dependence on wind speed is neglectable
    h_ref = 15  # reference height [m]
    rho_p = 2.2  # g cm^-3
    w = BT["BLH"] / BT["BLH"]
    U10 = np.abs(BT.U10.values)
    T = BT.T2M.values  # C
    P = BT.PS.values
    BLH = BT.BLH.values
    BLH = BLH.reshape(len(BLH), 1)
    vd, vs = deposition_velocity("default", Dp, rho_p, h_ref, U10, T, P)
    vd = vd.reshape(len(vd), 1)
    w_dry = 1 + 2 * h_ref / BLH * (np.exp(-vd * 3 * 3600 / 2 / h_ref) - 1)
    w[:] = np.squeeze(w_dry)
    return w


def bt_wet_depo(bt, scale=1.0, model="Wood"):
    # bt_ = bt.copy()
    hcloud = bt["BLH"] - bt["cloud_bas"]
    hcloud[hcloud < 0] = 0
    # compared to CL1 from ship the estimated cloud_bas sometimes much too low
    # this leads to hzi-->1 in cases where it should be <1!
    hzi = hcloud / bt["BLH"]
    I = bt["RTOT"] / 3  # rain rate in mm/hr
    I = (bt["RTOT"] + bt["SF"]) / 3  # rain + snowfall rate in mm/hr
    dt = 3 * 3600  # time step in seconds (3hr)
    if model == "Wood":
        lambda_INC = 2.25 * I / 3600 * hzi  # lambda in 1/sec # Zhen2018, Wood2004,2012
    elif model == "Flexpart":
        cl = 0.5 * np.power(I, 0.36)  # from felxpart paper it gets similar to Wood
        cl[cl == 0] = 1
        cl_flex10 = bt["tcLWC"] + bt["tcIWC"]
        cl_flex10[(cl_flex10 == 0)] = cl[(cl_flex10 == 0)]
        if model in ["Flexpart_6"]:
            lambda_INC = 0.9 * I / cl / 3600 * hzi
        elif model in ["Flexpart_10"]:
            lambda_INC = 0.9 * I / cl_flex10 / 3600 * hzi
            lambda_INC[lambda_INC < 0] = 0
            lambda_INC[(bt["tcLWC"] + bt["tcIWC"]) == 0] = 0
    else:
        print("not implementet")
    w = np.exp(-lambda_INC * dt * scale)
    bt_ = (bt[["timest_", "time", "BTindex"]]).copy()
    # bt_['w'] = w
    # bt_['W']=bt_.groupby(['timest_', 'BTindex']).aggregate('cumprod')['w']
    return w


def bt_accumulate_weights(bt, w, W_str="W"):
    W_depo = bt[["timest_", "BTindex", "time"]].copy()
    W_depo[W_str] = w.values
    W_depo[W_str] = W_depo.groupby(["timest_", "BTindex"]).aggregate("cumprod")[W_str]
    if 1:
        W_depo[W_str] = W_depo[W_str] / w.values  # remove first weight
        W_depo[W_str][w.values == 0.0] = 0.0
    return W_depo


def bt_get_values(bt_, return_value, return_times, aggr_trajs, aggr_time):
    # aggregates data over trajectories
    # option to calculated '', 'cumsum', or 'cumprod' along time axis
    # returns array of data shape (len(return_times), len(bt_.timest_) )
    timest_ = pd.DatetimeIndex(np.unique(bt_.timest_))
    bt_ = bt_[[return_value, "time", "timest_"]]
    if return_value in ["lat", "lon"]:  # special case for e.g. longitude
        bt_ = bt_.assign(sin=np.sin(bt_[return_value] / 180 * np.pi))
        bt_ = bt_.assign(cos=np.cos(bt_[return_value] / 180 * np.pi))
    bt_ = bt_.groupby(["timest_", "time"]).aggregate(
        aggr_trajs
    )  # average over the trajectories

    # build a wrapper for percentiles: (runs a few seconds!)
    # bt_ = bt_.groupby(['timest_', 'time']).aggregate(percentile(50))

    if len(aggr_time) > 0:
        bt_ = bt_.groupby("timest_").aggregate(aggr_time)  #

    if return_value in ["lat", "lon"]:  # special case for e.g. longitude
        bt_[return_value] = np.arctan2(bt_["sin"], bt_["cos"]) * 180 / np.pi

    if aggr_time in ["", "cumsum", "cumprod"]:
        x_ = []
        for return_time in return_times:
            x_.append(
                bt_.iloc[bt_.index.get_level_values("time") == np.float(return_time)][
                    return_value
                ].values
            )

        x_ = pd.DataFrame(np.transpose(x_), index=timest_)

    else:
        x_ = pd.DataFrame(
            bt_[return_value].values,
            columns=[return_value],
            index=pd.DatetimeIndex(np.unique(bt_.timest_)),
        )
        x_.index.rename("timest_", inplace=True)
    return x_


def bt_plot_values(bt_, return_value, return_times, aggr_trajs, aggr_time, Nhours=1):

    x_ = bt_get_values(bt_, return_value, return_times, aggr_trajs, aggr_time)

    if Nhours > 1:
        x_ = x_.resample(str(int(Nhours)) + "H").mean()

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(12, 4))
    ax = axes
    plt.pcolormesh(x_.index, return_times / 24, np.transpose(x_.values))
    cb = plt.colorbar(label=return_value)

    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_minor_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    ax.fmt_xdata = DateFormatter("%Y-%m-%d")
    fig.autofmt_xdate()

    ax.set_ylabel("air mass age [d]")

    leg_dates = [
        ["2016-12-20", "2017-01-21"],  # leg 1
        ["2017-01-22", "2017-02-25"],  # leg 2
        ["2017-02-26", "2017-03-19"],
    ]  # leg 3
    ax.set_xlim([pd.to_datetime(leg_dates[0][0]), pd.to_datetime(leg_dates[-1][1])])
    for leg_date in np.reshape(leg_dates, (6, 1)):
        ax.axvline(x=pd.to_datetime(leg_date), color="gray", linestyle="--")

    return fig, ax, cb


def corr_coef(x, y, method="pearson"):
    """
    Wrapper function around Pandas .corr method. To find correlation between two
    vectors/columns not necessarily in the same df.
    method accepts same arguments as .corr method -->
    method : {‘pearson’, ‘kendall’, ‘spearman’}
        pearson : standard correlation coefficient
        kendall : Kendall Tau correlation coefficient
        spearman : Spearman rank correlation
    """
    df = pd.DataFrame({"x": x, "y": y})
    r = df.corr(method=method).iloc[0, 1]
    return r


def bt_Npredict_time(btNpred, SSSF, params, W):  # W_depo):
    # btNpred['W']=(1-btNpred['SIF'])*(1-btNpred['LSM'])*btNpred['W_BLHP']*W_depo['W_depo']
    # F2N = 1/1000000*3*3600/btNpred['BLH']
    F2N = 1 / 1000000 * 3 * 3600 / btNpred["BLH"]
    if SSSF == "U10power":
        chi = params[0]
        F = np.power(btNpred["U10"].values, chi)
    else:
        # APS_range=np.array([1, 2.5])
        APS_RESOLUTION = (
            1 / 32
        )  # resolution of the APS used to get from bin center to bin edge

        Dca_l = aps_agg_meta.Dp_aps_low[0]  # um lowest bin
        Dca_h = aps_agg_meta.Dp_aps_high[
            0
        ]  # um to be defined, highest bin for integration of paramerisations
        APS_range = sssf.aps_DlowDhigh_to_range(Dca_l, Dca_h, RESOLUTION=APS_RESOLUTION)

        r80_RESOLUTION = (
            1 / 32
        )  # resolution at which the sssf are integrated to get total number flux
        r80_range = sssf.aps_D_to_r80(
            APS_range
        )  # conversion from Dp dry to r80 ! Discuss!
        r80 = np.power(
            10,
            np.arange(np.log10(r80_range[0]), np.log10(r80_range[1]), r80_RESOLUTION),
        )  # particle dry diameter in um

        U10 = btNpred["U10"].values
        SST = btNpred["SKT"].values - 273.15  # convert Kelvin to Celsius
        Re = btNpred["Re"].values
        _, F = sssf.sssf(sssf_str=SSSF, r80=r80, U10=U10, SST=SST, Re=Re)
        F[np.isnan(F)] = 1e-10  # set NaN values nearly zero to avoid spreading

    # btNpred['N'] = F*F2N*btNpred['W'] # contribution N[/cm3] = F[/m2/s]*dt[s]/BLH[m]
    btNpred["N"] = F * F2N * W  # contribution N[/cm3] = F[/m2/s]*dt[s]/BLH[m]

    btNpred = (
        btNpred[["timest_", "time", "N"]]
        .groupby(["timest_", "time"])
        .aggregate(np.mean)
    )  # average over the trajectories

    btNpred["N"] = btNpred.groupby("timest_")[
        "N"
    ].cumsum()  # for each timest_ calculate cumsum over the 'time'-axis
    return btNpred


def my_corr_time(y, X, qc, method):
    time2test = np.unique(X.index.get_level_values(1))  # get all "time" indicees
    corr = np.zeros_like(time2test) * np.NaN
    jtime = -1
    for testtime in time2test:
        jtime = jtime + 1
        x = X.iloc[X.index.get_level_values("time") == np.float(testtime)]
        x.reset_index(inplace=True)
        x.set_index(x.timest_, inplace=True)
        x = x.drop(columns=["timest_", "time"])
        if method in ["pearson", "spearman"]:
            corr[jtime] = corr_coef(
                np.squeeze(x[qc].values), y[qc].values, method=method
            )
        else:
            print("WRONG METHOD")
    return corr


def segments_corr_time(segments, X, method):
    time2test = np.unique(X.index.get_level_values(1))  # get all "time" indicees
    Nsegments = len(np.unique(segments["segment"].dropna()))
    corr = np.zeros([Nsegments, len(time2test)]) * np.NaN
    jtime = -1
    for testtime in time2test:
        jtime = jtime + 1
        x = X.iloc[X.index.get_level_values("time") == np.float(testtime)]
        x.reset_index(inplace=True)
        x.set_index(x.timest_, inplace=True)
        x = x.drop(columns=["timest_", "time"])
        for jseg in np.arange(0, Nsegments):  # np.unique(segments['segment']):
            qc = ~np.isnan(segments["y"]) & (segments["segment"] == jseg)
            y = segments["y"]
            if method in ["pearson", "spearman"]:
                corr[jseg, jtime] = corr_coef(
                    np.squeeze(x[qc].values), y[qc].values, method=method
                )
            else:
                print("WRONG METHOD")
    return corr


def segments_tau_opt(segments, X, method):
    corr = segments_corr_time(segments, X, method)
    time2test = np.unique(X.index.get_level_values(1))  # get all "time" indicees
    tau_opt = time2test[np.argmax(corr, axis=1)]
    return tau_opt, corr_opt
