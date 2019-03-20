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

    if SEP_METHOD is 'time':
        trn_ = trn.reset_index(drop=True)
        s1 = len(trn_)
        tst_ = tst.reset_index(drop=True)
        tst_.index = tst_.index +  s1

        leg_ = pd.concat([trn_, tst_], ignore_index=True)

        fig, ax = plt.subplots(1, 1, sharex=False, tight_layout=True, figsize=(12,6))
        ax.plot(trn_.index, y_tr_h, color='red', linewidth=2)
        ax.plot(tst_.index, y_ts_h, color='blue', linewidth=2)
    #    ax.plot(leg_whole_ts.index, leg_whole_ts.iloc[:,-1], color='green', linewidth=1)
        ax.plot(leg_.index, leg_, color='green', linewidth=1)
        ax.legend(['train prediction', 'test prediction', 'y ground truth' ])

    elif SEP_METHOD is 'random':

        trn_ = trn_.sort_index()
        tst_ = tst_.sort_index()

        leg_ = pd.concat([trn_, tst_], ignore_index=False)

        fig, ax = plt.subplots(1, 1, sharex=False, tight_layout=True, figsize=(12,6))
        ax.scatter(leg_.index, leg_, color='green', s=10, marker='o')
        ax.scatter(trn_.index, trn_, color='red', s=10, marker='x')
        ax.scatter(tst_.index, tst_, color='blue', s=15, marker='+') # [y_h]
    #    ax.plot(leg_whole_ts.index, leg_whole_ts.iloc[:,-1], color='green', linewidth=1)
        ax.legend(['y ground truth', 'train prediction', 'test prediction'])

    del trn_, tst_, leg_

    return fig, ax
##############################################################################################################
def aggregated_bins_regression_plot_weights(stats,sets,colnames,options,colors,SAVE=True):
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
        len_plot = len(options['LEG_P'])
    except TypeError:
        len_plot = 1
        options['LEG_P'] = [options['LEG_P']]

    bar_w = 0.1
    # print(savefigs)
    tickname = [str(aa) for aa in colnames]

    for sep_method in options['SEP_METHOD']:

        fig, ax = plt.subplots(len(sets), len_plot, sharey=False, tight_layout=False,
                               figsize=(5*len_plot,7), squeeze=False)

        for indbi, bin_ in enumerate(sets):
            index = np.arange(len(tickname))
            for ind, meth in enumerate(options['REGR_WITH_WEIGTHS']):
                for legind,leg in enumerate(options['LEG_P']):
                    string_plots = 'leg_' + str(leg) + '_' + sep_method + '_' + meth

                    if string_plots in stats[bin_]:
                        if 'weights' in  stats[bin_][string_plots]:
                            if meth == 'rbfgprard':
                                w = 1/(1 + 10*stats[bin_][string_plots]['weights'][0])#[string_plots + '_mean']
                                s = 1/(1 + 10*stats[bin_][string_plots]['weights'][1])#[string_plots + '_mean']
                            else:
                                w = stats[bin_][string_plots]['weights'][0]#[string_plots + '_mean']
                                s = stats[bin_][string_plots]['weights'][1]#[string_plots + '_mean']
                        else:
                            continue
                    else:
                        continue

                    ax[indbi,legind].bar(index, w, bar_w, color=tuple(colors[ind,:]))

                    ax[-1,legind].set_xticks(index)
                    if indbi == len(sets)-1:
                        ax[indbi,legind].set_xticklabels(tickname, rotation=90)
                    else:
                        ax[indbi,legind].set_xticklabels([])

                    ax[-1,legind].set_xlabel('LEG ' + str(leg), fontsize=16)

                index = index + bar_w

                # ax[-1,int(np.floor(len(options['LEG_P'])/2))].set_xticks(index)

            if ~indbi%2 | (indbi==0):
                ax[indbi,0].yaxis.set_label_position("left")
                ax[indbi,0].set_ylabel(bin_)
            else:
                ax[indbi,-1].yaxis.set_label_position("right")
                ax[indbi,-1].set_ylabel(bin_)


        index = np.arange(len(tickname))

        plt.legend(options['REGR_WITH_WEIGTHS'])
        plt.suptitle(sep_method + ' mode ')
        plt.show(block=False)#

        if SAVE:
            if sep_method == 'temporal_subsampling':
                add_string = '_' + options['SUB_OPTIONS']['submode']
            else:
                add_string = ''

        plt.savefig(options['SAVEFOLDER'] / ('weights_' + 'agg_bins' + '_leg_' + str(leg) + '_' + \
                 sep_method + '_' + meth + add_string + '.png'), bbox_inches='tight')

    return fig, ax

##############################################################################################################
def aggregated_bins_regression_plot_errors(stats,sets,options,colors,SAVE=True):
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
        len_plot = len(options['LEG_P'])
    except TypeError:
        len_plot = 1
        options['LEG_P'] = [options['LEG_P']]

    bar_w = 0.05
    for errmeasure in options['ERRMEASURE']:
        for sep_method in options['SEP_METHOD']:
            fig, ax = plt.subplots(2, len_plot, sharey=False,
                                   tight_layout=False, figsize=(5*len_plot,7), squeeze=False)

            for legind, leg in enumerate(options['LEG_P']):
                index = np.arange(1)
                for indbi, bin_ in enumerate(sets):
                    for ind, meth in enumerate(options['METHODS']):
                        string_plots =  'leg_' + str(leg) + '_' + sep_method + '_' + meth
                        if string_plots in stats[bin_]:
                            if errmeasure.lower() == 'rmse':
                                e_tr = stats[bin_][string_plots]['tr_RMSE'][0]
                                e_ts = stats[bin_][string_plots]['ts_RMSE'][0]
                                s_tr = stats[bin_][string_plots]['tr_RMSE'][1]
                                s_ts = stats[bin_][string_plots]['ts_RMSE'][1]

                            elif errmeasure.lower() == 'r2':
                                e_tr = stats[bin_][string_plots]['tr_R2'][0]
                                e_ts = stats[bin_][string_plots]['ts_R2'][0]
                                s_tr = stats[bin_][string_plots]['tr_R2'][1]
                                s_ts = stats[bin_][string_plots]['ts_R2'][1]
                        else:
                            print('wrong string')
                            e_tr = 0
                            e_ts = 0
                            s_tr = 0
                            s_ts = 0

                        ax[0, legind].bar(index, e_tr, bar_w, color=tuple(colors[ind,:]), yerr=s_tr)
                        ax[1, legind].bar(index, e_ts, bar_w, color=tuple(colors[ind,:]), yerr=s_ts)
                        index = index + bar_w

                    index = index + 2*bar_w
#                             print(string_plots)
#                             print('tst:' + str(e_ts))
#                             print('trn:' + str(e_tr))

                    if len(leg) == 1 or leg == 1:
                        ax[0,0].set_ylabel('training ' + errmeasure.upper())
                        ax[1,0].set_ylabel('testing ' + errmeasure.upper())

                    ax[1,legind].set_xlabel('LEG ' + str(leg), fontsize=16)

#                             ax[1,leg-1].set_xticks(index + len(options['METHODS'])*bar_w/2 - bar_w/2)
                    loc = [len(options['METHODS'])*bar_w/2 + ll - bar_w/2 for ll in (2*bar_w+bar_w*len(options['METHODS']))*np.arange(0,len(sets),1)]


                    if errmeasure.lower() == 'r2':
                        ax[0,legind].set_ylim([-0.5,1])
                        ax[0,legind].set_yticks(np.arange(-0.5,1,0.1))
                    ax[0,legind].set_xticks(loc)#np.arange(len(sets))/len(sets)+bar_w/2)
                    ax[0,legind].set_xticklabels('')
                    ax[0,legind].grid(color='black', which='both', axis='both', linestyle=':')

                    if errmeasure.lower() == 'r2':
                        ax[1,legind].set_ylim([-0.5,1])
                        ax[1,legind].set_yticks(np.arange(-0.5,1,0.1))
                    ax[1,legind].set_xticks(loc)#np.arange(len(sets))/len(sets)+bar_w/2)
                    ax[1,legind].set_xticklabels(sets, rotation=90)
                    ax[1,legind].grid(color='black', which='both', axis='both', linestyle=':')


            plt.legend(options['METHODS'])
            #plt.suptitle(sep_method + ' mode')
            plt.show(block=False)#


            if SAVE:
                if sep_method == 'temporal_subsampling':
                    add_string = '_' + options['SUB_OPTIONS']['submode']
                else:
                    add_string = ''


            plt.savefig(options['SAVEFOLDER'] / (errmeasure + '_leg_' + str(leg) + '_' + \
                            sep_method + add_string + '.png'), bbox_inches='tight')
    return fig, ax

##############################################################################################################
def single_bins_regression_plot_weights(stats,sets,colnames,options,colors,SAVE=True,ylim=[None, None]):
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
        len_plot = len(options['LEG_P'])
    except TypeError:
        len_plot = 1
        options['LEG_P'] = [options['LEG_P']]

    bar_w = 0.75
    for sep_method in options['SEP_METHOD']:
        tickname = colnames#options['COLNAMES']
        for ind, meth in enumerate(options['REGR_WITH_WEIGTHS']):
            fig, ax = plt.subplots(len(tickname), len_plot, sharey='row',
                            tight_layout=False,
                            figsize=(15,10), squeeze=False)

            for legind, leg in enumerate(options['LEG_P']):

                for ind_w, parname in enumerate(tickname):
                    index = np.arange(len(options['COLNAMES']))

                    string_plots = 'leg_' + str(leg) + '_' + sep_method + '_' + meth
                    w = []; s = []
                    for bin_ in sets:
                        w.append(np.squeeze(stats[bin_][string_plots]['weights'][0][ind_w]))#[0][ind_w])#[string_plots + '_mean']
                        s.append(np.squeeze(stats[bin_][string_plots]['weights'][1][ind_w]))#[ind_w] [string_plots + '_mean']

                    w = np.array(w)
                    s = np.array(s)

                    ax[ind_w,legind].bar(index, w, bar_w,
                                         color=tuple(colors[ind,:]), yerr=s)#olors[ind]'
                    # index = index + bar_w

                    # if leg == 1:
                    if ~ind_w%2 | (ind_w==0):
                        ax[ind_w,0].yaxis.set_label_position("left")
                        ax[ind_w,0].set_ylabel(parname,fontsize=6)
                    else:
                        ax[ind_w,-1].yaxis.set_label_position("right")
                        ax[ind_w,-1].set_ylabel(parname,fontsize=6)

                    for c in options['AGGREGATES']:
                        ax[ind_w,legind].axvline(c)

                index = np.arange(0,len(sets),1)
                ax[-1,legind].set_xticks(index[::5])
                ax[-1,legind].set_xticklabels(options['COLNAMES'][::5],fontsize=10,rotation='vertical')
                ax[ind_w,legind].grid(color='black', which='both', axis='y', linestyle=':')
                ax[-1,legind].set_ylim(ylim)

                plt.suptitle(meth + '_leg_' + str(leg) + '_' + sep_method)
                ax[-1,legind].set_xlabel('LEG ' + str(leg), fontsize=16)
                plt.show(block=False)#

            if SAVE:
                if sep_method == 'temporal_subsampling':
                    add_string = '_' + options['SUB_OPTIONS']['submode']
                else:
                    add_string = ''

            filename = 'weights_' + meth + '_leg_' + str(leg) + '_' + sep_method + '_' + meth + add_string
            plt.savefig((options['SAVEFOLDER'] / filename).with_suffix('.png'), bbox_inches='tight')

    return fig, ax

##############################################################################################################
def single_bins_regression_plot_errors(stats,sets,options,colors,SAVE=True):
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
        len_plot = len(options['LEG_P'])
    except TypeError:
        len_plot = 1
        options['LEG_P'] = [options['LEG_P']]

    bar_w = 0.45
    index = np.arange(len(options['ERRMEASURE']))
    for errmeasure in options['ERRMEASURE']:
        for sep_method in options['SEP_METHOD']:

            # tickname = dataset.subset_data_stack_variables([],
                    # varset, seatype=sea, mode='returnnames')
            index = np.arange(colors.shape[0])

            for ind, meth in enumerate(options['METHODS']):

                index = np.arange(len(sets))
                fig, [ax] = plt.subplots(1, len_plot, sharey=False, tight_layout=False,
                figsize=(len_plot*7,7), squeeze=False)

                print(len_plot)

                for legind,leg in enumerate(options['LEG_P']):
                    string_plots = 'leg_' + str(leg) + '_' + sep_method + '_' + meth
                    e_tr = []; e_ts = []
                    s_tr = []; s_ts = []

                    for bin_ in sets:
                        if string_plots not in stats[bin_].keys():
                            continue

                        if errmeasure.lower() == 'rmse':
                            e_tr.append(stats[bin_][string_plots]['tr_RMSE'][0])
                            e_ts.append(stats[bin_][string_plots]['ts_RMSE'][0])
                            s_tr.append(stats[bin_][string_plots]['tr_RMSE'][1])
                            s_ts.append(stats[bin_][string_plots]['ts_RMSE'][1])
                        elif errmeasure.lower() == 'r2':
                            e_tr.append(stats[bin_][string_plots]['tr_R2'][0])
                            e_ts.append(stats[bin_][string_plots]['ts_R2'][0])
                            s_tr.append(stats[bin_][string_plots]['tr_R2'][1])
                            s_ts.append(stats[bin_][string_plots]['ts_R2'][1])


                    l1 = ax[legind].plot(e_tr, color=tuple(colors[0,:]),label='train')#, yerr=s_tr)
                    l2 = ax[legind].plot(e_ts, color=tuple(colors[1,:]),label='test')#, yerr=s_tr) index+bar_w
                    e_tr = np.asarray(e_tr); s_tr = np.asarray(s_tr)
                    ax[legind].fill_between(range(len(e_tr)), e_tr-s_tr, e_tr+s_tr, alpha=0.3,                color=tuple(colors[0,:]))
                    e_ts = np.asarray(e_ts); s_ts = np.asarray(s_ts)
                    ax[legind].fill_between(range(len(e_ts)), e_ts-s_ts, e_ts+s_ts, alpha=0.3,                color=tuple(colors[1,:]))
#                         l1 = ax[leg-1].bar(index, e_ts, bar_w, color=tuple(colors[0,:]), yerr=s_ts)
#                         l2 = ax[leg-1].bar(index+bar_w, e_ts, bar_w, color=tuple(colors[1,:]), yerr=s_ts)

                    #if leg == 1:
                    ax[legind].set_ylabel(errmeasure, fontsize=20)
                    tic = [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
                    ax[legind].set_yticks(tic)    #[::5])
                    ax[legind].set_yticklabels([str(i) for i in tic],fontsize=15)
                    ax[legind].set_ylim([-0.21,1])

                    ax[legind].legend()

                    ax[legind].set_xticks(index)#[::5])
                    ax[legind].set_xticklabels(options['COLNAMES'], rotation='vertical', fontsize=15) #[::5],fontsize=10,rotation='vertical')

                    if errmeasure.lower() == 'r2':
                        ax[legind].grid(color='black', which='both', axis='both', linestyle=':')
                        # ax[legind].set_ylim([-0.5,1])

                    for c in options['AGGREGATES']:
                        ax[legind].axvline(c)

                    # plt.suptitle(errmeasure + '_' + meth + '_leg_' + str(leg) + '_' + sep_method)
                    fig.subplots_adjust(bottom=0.2) # or whatever
                    plt.show(block=False)#


                if SAVE:
                    if sep_method == 'temporal_subsampling':
                        add_string = '_' + options['SUB_OPTIONS']['submode']
                    else:
                        add_string = ''

                    filename = errmeasure + '_' + meth + '_leg_' + str(leg) + '_' + sep_method + add_string
                    plt.savefig((options['SAVEFOLDER'] / filename).with_suffix('.png'), bbox_inches='tight')

    return fig, ax

##############################################################################################################
def visualize_stereo_map(coordinates, values, min_va, max_va, markersize=75, fillconts='grey', fillsea='white', labplot='', plottype='scatter', make_caxes=True, cmap=plt.cm.viridis, set_in_ax=None):
    '''
        Visualize data on a polar stereographic projection map using cartopy on matplotlib.

        :param coordinates: give geographical coordinates of the points to plot
        :param values: can be a 1D vector of values (and a colormap is build accordingly) or it can be a 2D vector Nx3 corresponding to some colormap to be used for each datapoint
        :param fillconts: color to fill continents
        :param fillsea: color to fill seas
        :param labplot: the label for the data series, to use for legend and other handles
        :param min_va: min values to clip lower values (should be min of the series)
        :param max_va: max values to clip lower values (should be max of the series)
        :param plottype: (BETA) use different plotting tools (scatter or plot so far)
        :returns ax: figure handle.
        :returns ax: axes handles of the caropy map.
        :returns cbar: handle to the colorbar

        .. todo:: fix colors for plot as in scatter, but color lines rather than pointsself.
        .. todo:: add support for *custom* background image (e.g. sea surface temperature, wind magnitude, etc.) (use something.contourf() to interpolate linearly within a grid of values at known coordinates?)
        .. todo: add support for geo-unreferenced basemaps

        .. note:: The longitude lon_0 is at 6-o'clock, and the latitude circle boundinglat is tangent to the edge of the map at lon_0. Default value of lat_ts (latitude of true scale) is pole.
        .. note:: Latitude is in °N, longitude in is °E
    '''

    gps = coordinates.copy()
    # print(values.describe())

    tr1 = np.median(np.diff(coordinates.index.tolist()))
    tr2 = np.median(np.diff(values.index.tolist()))
    min_tres = max(tr1,tr2)
    min_tres = int(min_tres.total_seconds() / 60)

    # print(u"matching resolution @ %i minutes"%min_tres)
    values = pd.DataFrame(data=values.values, index=values.index, columns=['value'])
    values.index.name = 'timest_'

    coordinates = dataset.ts_aggregate_timebins(coordinates, time_bin=min_tres, operations={'': np.nanmedian}, index_position='middle')
    values = dataset.ts_aggregate_timebins(values, time_bin=min_tres, operations={'': np.nanmean}, index_position='middle')

    toplot = pd.concat((coordinates,values), axis=1)
    toplot = toplot.dropna()

    #
    # if coordinates.shape[0] != values.shape[0]:
    #     print('size of gps coordinates and variable to be plotted does not match, matching for you')

    # track = toplot.iloc[:,:2].copy()
    # toplot = toplot.dropna()

    # prepare basemap
    if set_in_ax is None:
        # fig = plt.gcf()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=ccrs.SouthPolarStereo())
    else:
        fig = set_in_ax[0]
        shax = set_in_ax[1]
        ax = fig.add_subplot(shax.shape[0],shax.shape[1],np.where(np.ravel(shax))[0][0]+1, projection=ccrs.SouthPolarStereo())

    # ax.set_proj
    ax.set_extent([-180, 180, -90, -35], ccrs.PlateCarree())
    ax.coastlines(linewidth=1.5)
    ax.add_feature(cfeature.LAND, facecolor=fillconts)
    ax.add_feature(cfeature.OCEAN, facecolor=fillsea)
    ax.gridlines(color='black', linestyle='--', alpha=0.5)
    #m.shadedrelief()
    # prepare colors
    # cmap = plt.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=min_va, vmax=max_va)
    colors = [cmap(normalize(value)) for value in toplot.iloc[:,-1]]

    # geo coord to plot coord
    # lon, lat = m(coordinates.iloc[:,1].values,coordinates.iloc[:,0].values)
    # map boat samples with values

    if plottype == 'scatter':
        ax.plot(gps.iloc[:,1], gps.iloc[:,0], transform=ccrs.Geodetic(), linewidth=1, color='black', zorder=1)
        ax.scatter(toplot.iloc[:,1].values,toplot.iloc[:,0].values, transform=ccrs.Geodetic(), c=np.squeeze(toplot.iloc[:,2].values),s=markersize, alpha=0.65, linewidth=0, label=labplot, cmap = cmap, zorder=2)
    elif plottype == 'plot':
        ax.plot(coordinates.iloc[:,1].values,coordinates.iloc[:,0].values,
            transform=ccrs.PlateCarree(),
            color=colors,s=markersize, linewidth=0, label=labplot)
        # im = m.plot(lon,lat,color=colors,linewidth=markersize,label=labplot)
    else:
        print('unrecognized plot')
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
    fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=False, tight_layout=True, figsize=(10,10))
    if len(color) > 1:
        if np.min(color) == 1:
            color -= 1
        labs = np.unique(color)
        s_color = plt.cm.Set1(labs)
        plot_color = plt.cm.Set1(color)

    row = 0; col = 0;
    for row, r_name in enumerate(df.columns):
        for col, c_name in enumerate(df.columns):
            if col == row:
                for ll in labs:
                    ax[row,col].hist(df.loc[color==ll,r_name],bins=nbins, color=s_color[ll,:], histtype='step')
            elif col != row:
                ax[row,col].scatter(df.iloc[:,col],df.iloc[:,row],c=plot_color, s=size, alpha=alpha)

            if col == 0:
                ax[row,col].set_ylabel(r_name)
            else:
                ax[row,col].set_ylabel('')

            if row == nrows-1:
                ax[row,col].set_xlabel(c_name)
            else:
                ax[row,col].set_xlabel('')

    return fig, ax

##############################################################################################################
def scatterplot_row(df, colname='', color=None, size=2):

    """
        Shorthand to plot a row of scatterplot (same as scatterplot_matrix but only a single row).

        :param df: dataframe contanining the data to be plotted, last column (default) or :param colname: versus all the others.
        :param color: color of the datapoint, blue default
        :param size: size of datapoints
        :returns: handles to figure and axes
    """

    # stirng check
    df.columns = [str(cc) for cc in df.columns]
    if not colname:
        colname = df.columns.tolist()[-1]

    nrows, ncols = df.shape[1], df.shape[1]
    fig, ax = plt.subplots(1, ncols, sharex=False, squeeze=False, sharey=False,
        tight_layout=True, figsize=(17,3))

    for col, c_name in enumerate(df.columns):
        if c_name == colname:
            ax[0,col].hist(df.loc[:,colname],bins=100, histtype='step')
        else:
            ax[0,col].scatter(df.iloc[:,col],df.loc[:,colname],c=color, s=size)

        if col == 0:
            ax[0,col].set_ylabel(colname)
        else:
            ax[0,col].set_ylabel('')

        ax[0,col].set_xlabel(c_name)

    return fig, ax

##############################################################################################################
def plot_binned_parameters_versus_averages(df, aerosol, subset_columns_parameters, nbins=25, range_par=[0,1]):

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

    f, ax = plt.subplots(2,len(subset_columns_parameters), squeeze=False, figsize=(10,3), sharex='col',
                sharey='row', gridspec_kw = {'height_ratios':[2, 1]}  )

    for vvnum, vv in enumerate(subset_columns_parameters):

        joind = df[vv].notnull() & aerosol.notnull()
        bins = np.linspace(np.percentile(df[vv].loc[joind],range_par[0]*100),
                np.percentile(df[vv].loc[joind],range_par[1]*100),
                nbins+1)

        bins_h = pd.cut(df[vv].loc[joind], bins, labels=[str(x) for x in range(nbins)], retbins=False)

        xi = aerosol.loc[joind].groupby(bins_h).agg(np.nanmean)
        st = aerosol.loc[joind].groupby(bins_h).agg(np.nanstd)

        ax[0,vvnum].errorbar(np.arange(nbins),
                        xi,
                        yerr=st,
                        ls='none', color='black')
        ax[0,vvnum].plot(np.arange(nbins), aerosol.loc[joind].groupby(bins_h).mean(), ls='-', color='red', linewidth=2)

        ax[1,vvnum].bar(np.arange(nbins),aerosol.loc[joind].groupby(bins_h).count())

        labels_ = ['{:.2f}'.format(xx) for xx in bins]

        if vvnum == 0:
            ax[0,vvnum].set_ylim(0,1.2*np.max(aerosol.loc[joind].groupby(bins_h).agg(np.nanmean)))

        ax[0,vvnum].set_ylabel('Aerosol value \n (mean per bin)')
        ax[0,vvnum].set_xticks(np.arange(0,nbins+1,10))
        ax[0,vvnum].set_xticklabels(labels_[::10])
        ax[0,vvnum].tick_params(labelbottom=True, labelleft=True)

        ax[1,vvnum].set_xlabel(vv)
        ax[1,vvnum].set_ylabel('Datapoint count')
        ax[1,vvnum].set_xticks(np.arange(0,nbins+1,10))
        ax[1,vvnum].set_xticklabels(labels_[::10])
        ax[1,vvnum].tick_params(labelbottom=True, labelleft=True)

    return f, ax
