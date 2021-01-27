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
import matplotlib as mpl

import pyantarctica.dataset as dataset
import matplotlib.pyplot as plt
import matplotlib.colorbar as clb

import cartopy.crs as ccrs
import cartopy.feature as cfeature

##############################################################################################################
def plot_predicted_timeseries(trn_, tst_, y_tr_h, y_ts_h, SEP_METHOD):
    """
        Plot time series of training and test data.

        .. todo:: probably needs a big update this function?

        :param trn_: dataframe of training data true labels
        :param tst_: dataframe of test data true labels
        :param y_tr_h: dataframe of training predicted labels
        :param y_ts_h: dataframe of test predicted labels
        :returns fig: handle of the figure
        :returns ax: handle of the axis
    """

    if SEP_METHOD is "time":
        trn_ = trn.reset_index(drop=True)
        s1 = len(trn_)
        tst_ = tst.reset_index(drop=True)
        tst_.index = tst_.index + s1

        leg_ = pd.concat([trn_, tst_], ignore_index=True)

        fig, ax = plt.subplots(1, 1, sharex=False, tight_layout=True, figsize=(12, 6))
        ax.plot(trn_.index, y_tr_h, color="red", linewidth=2)
        ax.plot(tst_.index, y_ts_h, color="blue", linewidth=2)
        #    ax.plot(leg_whole_ts.index, leg_whole_ts.iloc[:,-1], color='green', linewidth=1)
        ax.plot(leg_.index, leg_, color="green", linewidth=1)
        ax.legend(["train prediction", "test prediction", "y ground truth"])

    elif SEP_METHOD is "random":

        trn_ = trn_.sort_index()
        tst_ = tst_.sort_index()

        leg_ = pd.concat([trn_, tst_], ignore_index=False)

        fig, ax = plt.subplots(1, 1, sharex=False, tight_layout=True, figsize=(12, 6))
        ax.scatter(leg_.index, leg_, color="green", s=10, marker="o")
        ax.scatter(trn_.index, trn_, color="red", s=10, marker="x")
        ax.scatter(tst_.index, tst_, color="blue", s=15, marker="+")  #  [y_h]
        #    ax.plot(leg_whole_ts.index, leg_whole_ts.iloc[:,-1], color='green', linewidth=1)
        ax.legend(["y ground truth", "train prediction", "test prediction"])

    del trn_, tst_, leg_

    return fig, ax


##############################################################################################################
def aggregated_bins_regression_plot_weights(
    stats, sets, colnames, options, colors, SAVE=True
):
    """
        Plot weights associated to a linear regression model.

        .. todo:: probably needs a big update this function?

        :param stats: dictionary containing weights, as provided by the regression functions from the baseline_scripts module
        :param sets: particle bin aggregations
        :param options: same option dictionary provided to run_baselines functions (see help there)
        :param colors: array of RGBA values
        :param SAVE: boolean indicating wether to save or not the image, in the folder spefified in options.
        :returns: None
    """

    try:
        len_plot = len(options["LEG_P"])
    except TypeError:
        len_plot = 1
        options["LEG_P"] = [options["LEG_P"]]

    bar_w = 0.1
    # print(savefigs)
    tickname = [str(aa) for aa in colnames]

    for sep_method in options["SEP_METHOD"]:

        fig, ax = plt.subplots(
            len(sets),
            len_plot,
            sharey=False,
            tight_layout=False,
            figsize=(7 * len_plot, 7),
            squeeze=False,
        )

        for indbi, bin_ in enumerate(sets):
            for ind, meth in enumerate(options["REGR_WITH_WEIGTHS"]):
                for legind, leg in enumerate(options["LEG_P"]):
                    string_plots = "leg_" + str(leg) + "_" + sep_method + "_" + meth

                    if string_plots in stats[bin_]:
                        if "weights" in stats[bin_][string_plots]:
                            if meth == "rbfgprard":
                                w = 1 / (
                                    1 + 10 * stats[bin_][string_plots]["weights"][0]
                                )  # [string_plots + '_mean']
                                s = 1 / (
                                    1 + 10 * stats[bin_][string_plots]["weights"][1]
                                )  # [string_plots + '_mean']
                            else:
                                w = stats[bin_][string_plots]["weights"][
                                    0
                                ]  # [string_plots + '_mean']
                                s = stats[bin_][string_plots]["weights"][
                                    1
                                ]  # [string_plots + '_mean']
                        else:
                            continue
                    else:
                        continue

                    index = np.arange(len(w))

                    # print(indbi,legind,len(index),len(w),bar_w)
                    ax[indbi, legind].bar(index, w, bar_w, color=tuple(colors[ind, :]))

                    ax[indbi, legind].set_xticks(index)
                    # if indbi == len(sets)-1:
                    ax[indbi, legind].set_xticklabels(tickname, rotation=90)
                    # else:
                    # ax[indbi,legind].set_xticklabels([])

                    ax[-1, legind].set_xlabel("LEG " + str(leg), fontsize=16)

                index = index + bar_w

                # ax[-1,int(np.floor(len(options['LEG_P'])/2))].set_xticks(index)

            if ~indbi % 2 | (indbi == 0):
                ax[indbi, 0].yaxis.set_label_position("left")
                ax[indbi, 0].set_ylabel(bin_)
            else:
                ax[indbi, -1].yaxis.set_label_position("right")
                ax[indbi, -1].set_ylabel(bin_)

        index = np.arange(len(tickname))

        plt.legend(options["REGR_WITH_WEIGTHS"])
        plt.suptitle(sep_method + " mode ")
        plt.show(block=False)  #

        if SAVE:
            if sep_method == "temporal_subsampling":
                add_string = "_" + options["SUB_OPTIONS"]["submode"]
            else:
                add_string = ""

        plt.savefig(
            options["SAVEFOLDER"]
            / (
                "weights_"
                + "agg_bins"
                + "_leg_"
                + str(leg)
                + "_"
                + sep_method
                + "_"
                + meth
                + add_string
                + ".png"
            ),
            bbox_inches="tight",
        )

    return fig, ax


##############################################################################################################
def aggregated_bins_regression_plot_errors(stats, sets, options, colors, SAVE=True):
    """
        Plot errors RMSE or R2 (speficied in options['ERRMEASURE']) associated to a linear regression model

        .. todo:: probably needs a big update this function?

        :param stats: dictionary containing weights, as provided by the regression functions from the baseline_scripts module
        :param sets: particle bin aggregations
        :param options: same option dictionary provided to run_baselines functions (see help there)
        :param colors: array of RGBA values
        :param SAVE: boolean indicating wether to save or not the image, in the folder spefified in options.
        :returns: None
    """
    try:
        len_plot = len(options["LEG_P"])
    except TypeError:
        len_plot = 1
        options["LEG_P"] = [options["LEG_P"]]

    bar_w = 0.05
    for errmeasure in options["ERRMEASURE"]:
        for sep_method in options["SEP_METHOD"]:
            fig, ax = plt.subplots(
                2,
                len_plot,
                sharey=False,
                tight_layout=False,
                figsize=(5 * len_plot, 7),
                squeeze=False,
            )

            for legind, leg in enumerate(options["LEG_P"]):
                index = np.arange(1)
                for indbi, bin_ in enumerate(sets):
                    for ind, meth in enumerate(options["METHODS"]):
                        string_plots = "leg_" + str(leg) + "_" + sep_method + "_" + meth
                        if string_plots in stats[bin_]:
                            if errmeasure.lower() == "rmse":
                                e_tr = stats[bin_][string_plots]["tr_RMSE"][0]
                                e_ts = stats[bin_][string_plots]["ts_RMSE"][0]
                                s_tr = stats[bin_][string_plots]["tr_RMSE"][1]
                                s_ts = stats[bin_][string_plots]["ts_RMSE"][1]

                            elif errmeasure.lower() == "r2":
                                e_tr = stats[bin_][string_plots]["tr_R2"][0]
                                e_ts = stats[bin_][string_plots]["ts_R2"][0]
                                s_tr = stats[bin_][string_plots]["tr_R2"][1]
                                s_ts = stats[bin_][string_plots]["ts_R2"][1]
                        else:
                            print("wrong string")
                            e_tr = 0
                            e_ts = 0
                            s_tr = 0
                            s_ts = 0

                        ax[0, legind].bar(
                            index, e_tr, bar_w, color=tuple(colors[ind, :]), yerr=s_tr
                        )
                        ax[1, legind].bar(
                            index, e_ts, bar_w, color=tuple(colors[ind, :]), yerr=s_ts
                        )
                        index = index + bar_w

                    index = index + 2 * bar_w
                    #                             print(string_plots)
                    #                             print('tst:' + str(e_ts))
                    #                             print('trn:' + str(e_tr))

                    if len(leg) == 1 or leg == 1:
                        ax[0, 0].set_ylabel("training " + errmeasure.upper())
                        ax[1, 0].set_ylabel("testing " + errmeasure.upper())

                    ax[1, legind].set_xlabel("LEG " + str(leg), fontsize=16)

                    #                             ax[1,leg-1].set_xticks(index + len(options['METHODS'])*bar_w/2 - bar_w/2)
                    loc = [
                        len(options["METHODS"]) * bar_w / 2 + ll - bar_w / 2
                        for ll in (2 * bar_w + bar_w * len(options["METHODS"]))
                        * np.arange(0, len(sets), 1)
                    ]

                    if errmeasure.lower() == "r2":
                        ax[0, legind].set_ylim([-0.5, 1])
                        ax[0, legind].set_yticks(np.arange(-0.5, 1, 0.1))
                    ax[0, legind].set_xticks(
                        loc
                    )  # np.arange(len(sets))/len(sets)+bar_w/2)
                    ax[0, legind].set_xticklabels("")
                    ax[0, legind].grid(
                        color="black", which="both", axis="both", linestyle=":"
                    )

                    if errmeasure.lower() == "r2":
                        ax[1, legind].set_ylim([-0.5, 1])
                        ax[1, legind].set_yticks(np.arange(-0.5, 1, 0.1))
                    ax[1, legind].set_xticks(
                        loc
                    )  # np.arange(len(sets))/len(sets)+bar_w/2)
                    ax[1, legind].set_xticklabels(sets, rotation=90)
                    ax[1, legind].grid(
                        color="black", which="both", axis="both", linestyle=":"
                    )

            plt.legend(options["METHODS"])
            # plt.suptitle(sep_method + ' mode')
            plt.show(block=False)  #

            if SAVE:
                if sep_method == "temporal_subsampling":
                    add_string = "_" + options["SUB_OPTIONS"]["submode"]
                else:
                    add_string = ""

            plt.savefig(
                options["SAVEFOLDER"]
                / (
                    errmeasure
                    + "_leg_"
                    + str(leg)
                    + "_"
                    + sep_method
                    + add_string
                    + ".png"
                ),
                bbox_inches="tight",
            )
    return fig, ax


##############################################################################################################
def single_bins_regression_plot_weights(
    stats, sets, colnames, options, colors, SAVE=True, ylim=[None, None]
):
    """
        Plot weights associated to a linear regression modelself for every bin in a particle size distribution file.

        .. todo:: probably needs a big update this function?

        :param stats: dictionary containing weights, as provided by the regression functions from the baseline_scripts module
        :param sets: particle bin aggregations
        :param options: same option dictionary provided to run_baselines functions (see help there)
        :param colors: array of RGBA values
        :param SAVE: boolean indicating wether to save or not the image, in the folder spefified in options.
        :param ylim: speficy limits on y-axis. If both upper and lower limits are None, defaults to matplotlib standards
        :returns: None
    """
    try:
        len_plot = len(options["LEG_P"])
    except TypeError:
        len_plot = 1
        options["LEG_P"] = [options["LEG_P"]]

    bar_w = 0.75
    for sep_method in options["SEP_METHOD"]:
        tickname = colnames  # options['COLNAMES']
        for ind, meth in enumerate(options["REGR_WITH_WEIGTHS"]):
            fig, ax = plt.subplots(
                len(tickname),
                len_plot,
                sharey="row",
                tight_layout=False,
                figsize=(15, 10),
                squeeze=False,
            )

            for legind, leg in enumerate(options["LEG_P"]):

                for ind_w, parname in enumerate(tickname):
                    index = np.arange(len(options["COLNAMES"]))

                    string_plots = "leg_" + str(leg) + "_" + sep_method + "_" + meth
                    w = []
                    s = []
                    for bin_ in sets:
                        w.append(
                            np.squeeze(stats[bin_][string_plots]["weights"][0][ind_w])
                        )  # [0][ind_w])#[string_plots + '_mean']
                        s.append(
                            np.squeeze(stats[bin_][string_plots]["weights"][1][ind_w])
                        )  # [ind_w] [string_plots + '_mean']

                    w = np.array(w)
                    s = np.array(s)

                    ax[ind_w, legind].bar(
                        index, w, bar_w, color=tuple(colors[ind, :]), yerr=s
                    )  # olors[ind]'
                    # index = index + bar_w

                    # if leg == 1:
                    if ~ind_w % 2 | (ind_w == 0):
                        ax[ind_w, 0].yaxis.set_label_position("left")
                        ax[ind_w, 0].set_ylabel(parname, fontsize=6)
                    else:
                        ax[ind_w, -1].yaxis.set_label_position("right")
                        ax[ind_w, -1].set_ylabel(parname, fontsize=6)

                    for c in options["AGGREGATES"]:
                        ax[ind_w, legind].axvline(c)

                index = np.arange(0, len(sets), 1)
                ax[-1, legind].set_xticks(index[::5])
                ax[-1, legind].set_xticklabels(
                    options["COLNAMES"][::5], fontsize=10, rotation="vertical"
                )
                ax[ind_w, legind].grid(
                    color="black", which="both", axis="y", linestyle=":"
                )
                ax[-1, legind].set_ylim(ylim)

                plt.suptitle(meth + "_leg_" + str(leg) + "_" + sep_method)
                ax[-1, legind].set_xlabel("LEG " + str(leg), fontsize=16)
                plt.show(block=False)  #

            if SAVE:
                if sep_method == "temporal_subsampling":
                    add_string = "_" + options["SUB_OPTIONS"]["submode"]
                else:
                    add_string = ""

            filename = (
                "weights_"
                + meth
                + "_leg_"
                + str(leg)
                + "_"
                + sep_method
                + "_"
                + meth
                + add_string
            )
            plt.savefig(
                (options["SAVEFOLDER"] / filename).with_suffix(".png"),
                bbox_inches="tight",
            )

    return fig, ax


##############################################################################################################
def single_bins_regression_plot_errors(stats, sets, options, colors, SAVE=True):
    """
        Plot errors (as specified in options['ERRMEASURE']) associated to a linear regression modelself for every bin in a particle size distribution file.

        .. todo:: probably needs a big update this function?

        :param stats: dictionary containing weights, as provided by the regression functions from the baseline_scripts module
        :param sets: particle bin aggregations
        :param options: same option dictionary provided to run_baselines functions (see help there)
        :param colors: array of RGBA values
        :param SAVE: boolean indicating wether to save or not the image, in the folder spefified in options.
        :returns: None
    """

    try:
        len_plot = len(options["LEG_P"])
    except TypeError:
        len_plot = 1
        options["LEG_P"] = [options["LEG_P"]]

    bar_w = 0.45
    index = np.arange(len(options["ERRMEASURE"]))
    for errmeasure in options["ERRMEASURE"]:
        for sep_method in options["SEP_METHOD"]:

            # tickname = dataset.subset_data_stack_variables([],
            # varset, seatype=sea, mode='returnnames')
            index = np.arange(colors.shape[0])

            for ind, meth in enumerate(options["METHODS"]):

                index = np.arange(len(sets))
                fig, [ax] = plt.subplots(
                    1,
                    len_plot,
                    sharey=False,
                    tight_layout=False,
                    figsize=(len_plot * 9, 7),
                    squeeze=False,
                )

                # print(len_plot)

                for legind, leg in enumerate(options["LEG_P"]):
                    string_plots = "leg_" + str(leg) + "_" + sep_method + "_" + meth
                    e_tr = []
                    e_ts = []
                    s_tr = []
                    s_ts = []

                    for bin_ in sets:
                        if string_plots not in stats[bin_].keys():
                            continue

                        if errmeasure.lower() == "rmse":
                            e_tr.append(stats[bin_][string_plots]["tr_RMSE"][0])
                            e_ts.append(stats[bin_][string_plots]["ts_RMSE"][0])
                            s_tr.append(stats[bin_][string_plots]["tr_RMSE"][1])
                            s_ts.append(stats[bin_][string_plots]["ts_RMSE"][1])
                        elif errmeasure.lower() == "r2":
                            e_tr.append(stats[bin_][string_plots]["tr_R2"][0])
                            e_ts.append(stats[bin_][string_plots]["ts_R2"][0])
                            s_tr.append(stats[bin_][string_plots]["tr_R2"][1])
                            s_ts.append(stats[bin_][string_plots]["ts_R2"][1])

                    e_tr = np.asarray(e_tr)
                    s_tr = np.asarray(s_tr)
                    e_ts = np.asarray(e_ts)
                    s_ts = np.asarray(s_ts)

                    if 0:
                        l1 = ax[legind].plot(
                            e_tr, color=tuple(colors[0, :]), label="train"
                        )  # , yerr=s_tr)
                        l2 = ax[legind].plot(
                            e_ts, color=tuple(colors[1, :]), label="test"
                        )  # , yerr=s_tr) index+bar_w
                        ax[legind].fill_between(
                            range(len(e_tr)),
                            e_tr - s_tr,
                            e_tr + s_tr,
                            alpha=0.3,
                            color=tuple(colors[0, :]),
                        )
                        ax[legind].fill_between(
                            range(len(e_ts)),
                            e_ts - s_ts,
                            e_ts + s_ts,
                            alpha=0.3,
                            color=tuple(colors[1, :]),
                        )
                    else:
                        l1 = ax[leg - 1].bar(
                            index - bar_w / 4,
                            e_tr,
                            width=bar_w / 2,
                            yerr=s_tr,
                            color=tuple(colors[0, :]),
                            label="train",
                        )
                        l2 = ax[leg - 1].bar(
                            index + bar_w / 4,
                            e_ts,
                            width=bar_w / 2,
                            yerr=s_ts,
                            color=tuple(colors[1, :]),
                            label="test",
                        )

                    # if leg == 1:
                    ax[legind].set_ylabel(errmeasure, fontsize=20)
                    # tic = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
                    if errmeasure.lower() == "r2":
                        tic = [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
                        ax[legind].set_yticks(tic)  # [::5])
                        ax[legind].set_yticklabels([str(i) for i in tic], fontsize=12)
                        ax[legind].set_ylim([-0.21, 1])

                    ax[legind].legend(loc="upper left")

                    ax[legind].set_xticks(index)  # [::5])
                    ax[legind].set_xticklabels(
                        options["COLNAMES"], rotation="vertical", fontsize=12
                    )  # [::5],fontsize=10,rotation='vertical')

                    # if errmeasure.lower() == 'r2':
                    ax[legind].grid(
                        color="black", which="both", axis="both", linestyle=":"
                    )
                    #  ax[legind].set_ylim([-0.5,1])

                    for c in options["AGGREGATES"]:
                        ax[legind].axvline(c)

                    #  plt.suptitle(errmeasure + '_' + meth + '_leg_' + str(leg) + '_' + sep_method)
                    fig.subplots_adjust(bottom=0.2)  # or whatever
                    plt.show(block=False)  #

                if SAVE:
                    if sep_method == "temporal_subsampling":
                        add_string = "_" + options["SUB_OPTIONS"]["submode"]
                    else:
                        add_string = ""

                    filename = (
                        errmeasure
                        + "_"
                        + meth
                        + "_leg_"
                        + str(leg)
                        + "_"
                        + sep_method
                        + add_string
                    )
                    plt.savefig(
                        (options["SAVEFOLDER"] / (filename + "_TEST_BARS")).with_suffix(
                            ".pdf"
                        ),
                        bbox_inches="tight",
                    )

    return fig, ax


##############################################################################################################
def visualize_stereo_map(
    coordinates,
    values,
    min_va,
    max_va,
    markersize=75,
    fillconts="grey",
    fillsea="white",
    labplot="",
    plottype="scatter",
    make_caxes=True,
    cmap=plt.cm.viridis,
    set_in_ax=None,
    centercm=False,
    resample_time=None,
    markeralpha=0.75,
    markerframe=False,
):
    """
        Visualize data on a polar stereographic projection map using cartopy on matplotlib.

        :param coordinates: give geographical coordinates of the points to plot
        :param values: can be a 1D vector of values (and a colormap is build accordingly) or it can be a 2D vector Nx3 corresponding to some colormap to be used for each datapoint
        :param fillconts: color to fill continents
        :param fillsea: color to fill seas
        :param labplot: the label for the data series, to use for legend and other handles
        :param min_va: min values to clip lower values (should be min of the series)
        :param max_va: max values to clip lower values (should be max of the series)
        :param plottype: (BETA) use different plotting tools (scatter or plot so far)
        :param markeralpha: transparency of the marker defaults to 0.75
        :param markerframe: adds a frame around the marker with width 1.2*markersize (only recomended with markeralpha=1), defaults to False
        :returns ax: figure handle.
        :returns ax: axes handles of the caropy map.
        :returns cbar: handle to the colorbar

        .. todo:: fix colors for plot as in scatter, but color lines rather than pointsself.
        .. todo:: add support for *custom* background image (e.g. sea surface temperature, wind magnitude, etc.) (use something.contourf() to interpolate linearly within a grid of values at known coordinates?)
        .. todo: add support for geo-unreferenced basemaps

        .. note:: The longitude lon_0 is at 6-o'clock, and the latitude circle boundinglat is tangent to the edge of the map at lon_0. Default value of lat_ts (latitude of true scale) is pole.
        .. note:: Latitude is in °N, longitude in is °E
    """
    from matplotlib.colors import ListedColormap
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    from matplotlib.path import Path as mpath

    gps = coordinates.copy()
    # print(values.describe())

    tr1 = np.median(np.diff(coordinates.index.tolist()))
    tr2 = np.median(np.diff(values.index.tolist()))
    min_tres = max(tr1, tr2)
    min_tres = int(min_tres.total_seconds() / 60)

    # print(u"matching resolution @ %i minutes"%min_tres)
    values = pd.DataFrame(data=values.values, index=values.index, columns=["value"])
    values.index.name = "timest_"

    coordinates = dataset.ts_aggregate_timebins(
        coordinates,
        time_bin=min_tres,
        operations={"": np.nanmedian},
        index_position="middle",
    )
    values = dataset.ts_aggregate_timebins(
        values, time_bin=min_tres, operations={"": np.nanmean}, index_position="middle"
    )
    # values[values > max_va] = max_va
    # values[values < min_va] = min_va

    if resample_time == None:
        resample_time = min_tres

    toplot = pd.concat((coordinates, values), axis=1)
    toplot = dataset.ts_aggregate_timebins(
        toplot,
        time_bin=resample_time,
        operations={"": np.nanmedian},
        index_position="middle",
    )
    toplot = toplot.dropna()

    ortho = ccrs.SouthPolarStereo()  #

    # ortho = ccrs.Orthographic(central_longitude=0, central_latitude=-90)
    geo = ccrs.Geodetic()
    geo_scatter = ccrs.PlateCarree() # SL > added this to faciliate the scatter plot for Cartopy==0.18.0 
    # ortho = ccrs.Orthographic(central_longitude=0, central_latitude=-90)
    # geo = ccrs.Geodetic()
    # prepare basemap
    if set_in_ax is None:
        # fig = plt.gcf()
        # ortho = ccrs.Orthographic(central_longitude=0, central_latitude=-90)
        # geo = ccrs.Geodetic()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ortho)
    else:
        fig = set_in_ax[0]
        shax = set_in_ax[1]
        ax = fig.add_subplot(
            shax.shape[0],
            shax.shape[1],
            np.where(np.ravel(shax))[0][0] + 1,
            projection=ortho,
        )

    ax.set_extent([-180, 180, -90, -35], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor=fillconts, zorder=2)
    ax.add_feature(cfeature.OCEAN, facecolor=fillsea, zorder=1)
    ax.coastlines(linewidth=1.5, zorder=2)
    # ax.gridlines(color='black', linestyle='--', alpha=0.5)
    # m.shadedrelief()
    # prepare colors

    if not hasattr(cmap, "shape"):
        if centercm:
            th_ = -np.max((np.abs(min_va), max_va)), np.max((np.abs(min_va), max_va))
        else:
            th_ = min_va, max_va

            normalize = mpl.colors.Normalize(vmin=th_[0], vmax=th_[1])

        # cmap = mpl.cm.LinearSegmentedColormap(
        #     [cmap(normalize(value)) for value in toplot.iloc[:, -1]]
        # )

    else:
        cmap = ListedColormap(cmap)

    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

    if plottype == "scatter":
        ax.plot(
            gps.iloc[:, 1],
            gps.iloc[:, 0],
            transform=geo,
            linewidth=1,
            color="black",
            zorder=1+1,
        )
        
        if markerframe:
            ax.scatter(
                toplot.iloc[:, 1].values,
                toplot.iloc[:, 0].values,
                transform=geo_scatter, # geo, # SL > changed here this to faciliate the scatter plot for Cartopy==0.18.0
                s=markersize*1.2,
                color="black",
                alpha=markeralpha,
                zorder=1+1,
            )  #  , norm=norm

        ax.scatter(
            toplot.iloc[:, 1].values,
            toplot.iloc[:, 0].values,
            transform=geo_scatter, # geo, # SL > changed here this to faciliate the scatter plot for Cartopy==0.18.0
            c=toplot.iloc[:, 2].values,
            s=markersize,
            alpha=markeralpha, # added option to change the marker transparency
            linewidth=0,
            label=labplot,
            cmap=cmap,vmin=min_va, vmax=max_va, # SL > added here to actually use the min max values for the color scale!
            zorder=2+1,
        )  #  , norm=norm

        # theta = np.linspace(0, 2*np.pi, 100)
        # center, radius = [0.5, 0.5], 0.5
        # verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        # circle = mpath(verts * radius + center)
        # ax.set_boundary(circle, transform=ax.transAxes)

        ax.gridlines(
            draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="-", zorder=2,
        )  # crs=ccrs.PlateCarree(),

        # Define gridline locations and draw the lines using cartopy's built-in gridliner:
        # xticks = [-110, -50, -40, -30, -20, -11, 0, 10, 20, 30, 40, 50]
        # yticks = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        # ax.gridlines(xlocs=xticks, ylocs=yticks)

        # Label the end-points of the gridlines using the custom tick makers:
        # ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        # ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
        # lambert_xticks(ax, xticks)
        # lambert_yticks(ax, yticks)

        # ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=ccrs.PlateCarree())
        # ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
        # lon_formatter = LongitudeFormatter(zero_direction_label=True)
        # lat_formatter = LatitudeFormatter()
        # ax.xaxis.set_major_formatter(lon_formatter)
        # ax.yaxis.set_major_formatter(lat_formatter)
        # import matplotlib.ticker as mticker # grid lines
        # gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
        #               linewidth=0.5, color='gray', alpha=0.5, linestyle='-')
        # gl.ylocator = mticker.FixedLocator(np.array([-180, 180, -90, -35]))
        # gl.xlocator = mticker.FixedLocator(np.linspace(0,360,13,dtype=int))

        # ax.gridlines(crs = ccrs.PlateCarree(),color='lightgrey', linestyle='-', draw_labels=True)

    elif plottype == "plot":
        ax.plot(
            coordinates.iloc[:, 1].values,
            coordinates.iloc[:, 0].values,
            transform=ccrs.PlateCarree(),
            color=cmap,
            s=markersize,
            linewidth=0,
            label=labplot,
            zorder=2+1,
        )
        # im = m.plot(lon,lat,color=colors,linewidth=markersize,label=labplot)
    else:
        print("unrecognized plot")
        return -1

    # ax.set_title(labplot,fontsize=35)
    if make_caxes:
        cax, _ = clb.make_axes(ax)
        cbar = clb.ColorbarBase(cax, cmap=cmap, norm=normalize)
        return fig, ax, cbar
    else:
        return fig, ax


##############################################################################################################
def scatterplot_matrix(df, color=None, size=2, nbins=100, alpha=0.5):

    """
        Shorthand to plot a matrix of scatterplot.

        :param df: dataframe contanining the data to be plotted, all columns versus all the others.
        :param color: color of the datapoint, blue default
        :param size: size of datapoints
        :returns: handles to figure and axes
    """

    df.columns = [str(cc) for cc in df.columns]

    nrows, ncols = df.shape[1], df.shape[1]
    fig, ax = plt.subplots(
        nrows, ncols, sharex=False, sharey=False, tight_layout=True, figsize=(10, 10)
    )
    if len(color) > 1:
        if np.min(color) == 1:
            color -= 1
        labs = np.unique(color)
        s_color = plt.cm.Set1(labs)
        plot_color = plt.cm.Set1(color)

    row = 0
    col = 0
    for row, r_name in enumerate(df.columns):
        for col, c_name in enumerate(df.columns):
            if col == row:
                for ll in labs:
                    ax[row, col].hist(
                        df.loc[color == ll, r_name],
                        bins=nbins,
                        color=s_color[ll, :],
                        histtype="step",
                    )
            elif col != row:
                ax[row, col].scatter(
                    df.iloc[:, col], df.iloc[:, row], c=plot_color, s=size, alpha=alpha
                )

            if col == 0:
                ax[row, col].set_ylabel(r_name)
            else:
                ax[row, col].set_ylabel("")

            if row == nrows - 1:
                ax[row, col].set_xlabel(c_name)
            else:
                ax[row, col].set_xlabel("")

    return fig, ax


##############################################################################################################
def scatterplot_row(df, colname="", color="blue", size=2, nbins=50, labels=""):

    """
        Shorthand to plot a row of scatterplot (same as scatterplot_matrix but only a single row).

        :param df: dataframe contanining the data to be plotted, last column (default) or :param colname: versus all the others.
        :param color: color of the datapoint, blue default. Can be a column of the matrix
        :param size: size of datapoints
        :param nbins: number of bins for the histogram
        :param labels: list of labels for the axes
        :returns: handles to figure and axes
    """
    from matplotlib import rc

    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc("text", usetex=True)

    df.columns = [str(cc) for cc in df.columns]
    if not colname:
        colname = df.columns.tolist()[-1]
    if not labels:
        labels = df.columns.tolist()

    if color in df.columns.tolist():
        col_list = df.columns.tolist()
        col_list.remove(color)

        ncols = len(col_list)
        fig, ax = plt.subplots(
            1,
            ncols,
            sharex=False,
            squeeze=True,
            sharey=False,
            tight_layout=True,
            figsize=(25, 4),
        )

        for col, c_name in enumerate(col_list):
            if c_name == colname:
                for icol in np.unique(df[color]):
                    ax[col].hist(
                        df.loc[df[color] == icol, colname].dropna(),
                        bins=nbins,
                        histtype="step",
                        linewidth=2.5,
                    )
            else:
                subdf = df[[colname, c_name, color]].dropna()
                for icol in np.unique(df[color]):
                    ax[col].scatter(
                        subdf.loc[subdf[color] == icol, c_name],
                        subdf.loc[subdf[color] == icol, colname],
                        s=size,
                        alpha=0.7,
                    )

            if col == 0:
                ax[col].set_ylabel(labels[col])
            else:
                ax[col].set_ylabel("")

            ax[col].set_xlabel(labels[col])
    else:
        color = None
        ncols = len(df.columns)
        fig, ax = plt.subplots(
            1,
            ncols,
            sharex=False,
            squeeze=True,
            sharey=False,
            tight_layout=True,
            figsize=(25, 4),
        )
        for col, c_name in enumerate(df.columns):
            if c_name == colname:
                ax[col].hist(df.loc[:, colname].dropna(), bins=nbins, histtype="step")
            else:
                subdf = df[[colname, c_name]].dropna()
                ax[col].scatter(
                    subdf.loc[:, c_name], subdf.loc[:, colname], c=color, s=size
                )

            if col == 0:
                ax[col].set_ylabel(labels[col])
            else:
                ax[col].set_ylabel("")

            ax[col].set_xlabel(labels[col])
    return fig, ax


##############################################################################################################
def plot_binned_parameters_versus_averages(
    df, aerosol, subset_columns_parameters, nbins=25, range_par=[0, 1]
):

    """
        Plot binned parameters speficied in a dataframe (mean of bins) versus a corresponding mean value of aerosol concentration.

        :param df: dataframe contanining the parameter data to be plotted
        :type df: dataframe of floats
        :param aerosol: dataframe of matching aerosol data observations
        :type aerosol:
        :param subset_columns_parameters: subset of the columns from the df used to plot the binned data
        :param nbins: integer number of bins to use when splitting data
        :param range_par: (2,)-sized floats in [0,1] specific the lower and upper percentiles of the parameters data, used as limits on the x-axis before binning.
        :returns: handles to figure and axes
    """

    f, ax = plt.subplots(
        2,
        len(subset_columns_parameters),
        squeeze=False,
        figsize=(10, 3),
        sharex="col",
        sharey="row",
        gridspec_kw={"height_ratios": [2, 1]},
        tight_layout=True,
    )

    for vvnum, vv in enumerate(subset_columns_parameters):

        joind = df[vv].notnull() & aerosol.notnull()
        bins = np.linspace(
            np.percentile(df[vv].loc[joind], range_par[0] * 100),
            np.percentile(df[vv].loc[joind], range_par[1] * 100),
            nbins + 1,
        )

        bins_h = pd.cut(
            df[vv].loc[joind],
            bins,
            labels=[str(x) for x in range(nbins)],
            retbins=False,
        )

        xi = aerosol.loc[joind].groupby(bins_h).agg(np.nanmean)
        st = aerosol.loc[joind].groupby(bins_h).agg(np.nanstd)

        ax[0, vvnum].errorbar(np.arange(nbins), xi, yerr=st, ls="none", color="black")
        ax[0, vvnum].plot(
            np.arange(nbins),
            aerosol.loc[joind].groupby(bins_h).mean(),
            ls="-",
            color="red",
            linewidth=2,
        )

        ax[1, vvnum].bar(np.arange(nbins), aerosol.loc[joind].groupby(bins_h).count())

        labels_ = ["{:.2f}".format(xx) for xx in bins]

        if vvnum == 0:
            ax[0, vvnum].set_ylim(
                0, 1.75 * np.max(aerosol.loc[joind].groupby(bins_h).agg(np.nanmean))
            )

        ax[0, vvnum].set_ylabel("Aerosol value \n (mean per bin)")
        ax[0, vvnum].set_xticks(np.arange(0, nbins + 1, 10))
        ax[0, vvnum].set_xticklabels(labels_[::10])
        ax[0, vvnum].tick_params(labelbottom=True, labelleft=True)

        ax[1, vvnum].set_xlabel(vv)
        ax[1, vvnum].set_ylabel("Datapoint count")
        ax[1, vvnum].set_xticks(np.arange(0, nbins + 1, 10))
        ax[1, vvnum].set_xticklabels(labels_[::10])
        ax[1, vvnum].tick_params(labelbottom=True, labelleft=True)

        # ax[0,vvnum].autoscale(enable=True, axis='x', tight=True)
        # ax[1,vvnum].autoscale(enable=True, axis='x', tight=True)

    return f, ax


# ########################
def interactive_map(v1, options):
    """
        Creates and interactive map of the variable v1 on the cruise track contained in options['gps_file']

        :param v1: dataframe / series containing the variable to be visualized
        :param options: 
            - options['plot_size'] : global figure scaling (make all of them larger of smaller by this multiplier), except the interactive plots\n",
            - options['figsize'] : width, height for the interactive plot\n",
            - options['scatter_markersize'] :  size of the markers in the scatterplot\n",
            - options['map_scatter_markersize'] : size of the markers in the static geographical map\n",
            - options['map_temporal_aggregation'] : Hours to aggregate in the static and interactive geographical map.\n",
            - options['resampling_operation'] : resample points on map temporally,
            - options['colormap'] : colormap from plt.cm
            - options['gps_file'] : file containing the gps coordinates

        :returns: interactive figure object
    """
    
    import cartopy.crs as ccrs
    import holoviews as hv
    from holoviews import opts, dim
    import geoviews as gv
    import geoviews.feature as gf
    import simplekml

    hv.extension("bokeh", "matplotlib")

    vname = v1.columns.tolist()[0]

    stretch = [0, 100]

    tres = np.median(np.diff(v1.index.tolist()))
    tres = 1 * int(tres.total_seconds() / 60)
    # vf1 = dataset.ts_aggregate_timebins(v1.to_frame(), time_bin=tres, operations={'': np.nanmean}, index_position='middle')

    #  come up with leg coloring
    leg_series = dataset.add_legs_index(v1)["leg"]

    vcol = dataset.ts_aggregate_timebins(
        v1,
        time_bin=tres,
        operations={"": options["resampling_operation"]},
        index_position="initial",
    )

    vcol.columns = ["color"]
    min_ = np.percentile(vcol.dropna(), stretch[0])
    max_ = np.percentile(vcol.dropna(), stretch[1])
    # print(min_,max_)
    vcol[vcol < min_] = min_
    vcol[vcol > max_] = max_

    coordinates_raw = dataset.read_standard_dataframe(options["gps_file"])[
        ["latitude", "longitude"]
    ]

    # mode_ = lambda x : stats.mode(x)[0]
    #  Deal with coordinates
    coordinates = dataset.ts_aggregate_timebins(
        coordinates_raw,
        time_bin=int(np.floor(options["map_temporal_aggregation"] * 60)),
        operations={"": np.nanmedian},
        index_position="initial",
    )

    #  Resample and merge coordinates + data
    to_plot = pd.merge(coordinates, v1, left_index=True, right_index=True)
    to_plot = pd.merge(
        to_plot,
        pd.DataFrame(
            data=to_plot.index.tolist(), columns=["date"], index=to_plot.index
        ),
        left_index=True,
        right_index=True,
    )
    to_plot = to_plot.dropna()

    if options["kml_file"]:
        kml_ = simplekml.Kml()
        fol = kml_.newfolder(name="LV_KML")

        colz = (to_plot.loc[:, vname] - to_plot.loc[:, vname].min()) / (
            to_plot.loc[:, vname].max() - to_plot.loc[:, vname].min()
        )
        colz = np.floor(255 * plt.cm.Spectral_r(colz.values)).astype(int)
        c = 0
        for lat, lon, val, date in to_plot.values:
            #     print(lat, lon, val, date)
            #     print(row[1].loc['LV#11'])
            pnt = fol.newpoint(name=str(date), coords=[(lon, lat)])
            pnt.style.iconstyle.icon.href = (
                "http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png"
            )
            pnt.style.iconstyle.scale = 1  # Icon thrice as big
            pnt.style.iconstyle.color = simplekml.Color.rgb(
                colz[c, 0], colz[c, 1], colz[c, 2], 255
            )
            pnt.style.labelstyle.scale = 0.5
            c += 1

        kml_.save(options["kml_file"])

    #  colorbar limits
    min_v1 = min_  # np.percentile(to_plot.loc[:,vname].dropna(), stretch[0])
    max_v1 = max_  # np.percentile(to_plot.loc[:,vname].dropna(), stretch[1])

    #  Create geoviews datasets
    v1_tomap = hv.Dataset(
        to_plot.loc[:, ["longitude", "latitude", "date", vname]],
        kdims=["longitude", "latitude"],
        vdims=[hv.Dimension(vname, range=(min_v1, max_v1))],
        group=vname,
    )

    points_v1 = v1_tomap.to(gv.Points, kdims=["longitude", "latitude"])

    gps_track = gv.Dataset(coordinates_raw)
    track = gv.Path(gps_track, kdims=["longitude", "latitude"])
    # land_ = gf.land#.options(facecolor='red')

    point_map_v1 = points_v1.opts(
        projection=ccrs.SouthPolarStereo(),
        cmap=options["colormap"],
        size=5,
        tools=["hover"],  #  ['hover'],
        width=500,
        height=400,
        color_index=2,
        colorbar=True,
    )

    track_map = track.opts(projection=ccrs.SouthPolarStereo()).opts(color="black")

    return (gf.land * gf.coastline * track_map * point_map_v1).opts(
        title=vname, width=options["figsize"][0], height=options["figsize"][1]
    )
