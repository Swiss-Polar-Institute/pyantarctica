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

# import mpl_toolkits.basemap as carto
##############################################################################################################
def plot_predicted_timeseries(trn_, tst_, y_ts_h, y_tr_h, SEP_METHOD):
    # Should probably be moved in a visualization module, here not really modeling stuff...

    if SEP_METHOD is 'time':
        # %matplotlib qt
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

        #trn_ = trn.copy()
        #tst_ = tst.copy()
        #trn_['y_h'] = y_tr_h
        #tst_['y_h'] = y_ts_h

        trn_ = trn_.sort_index()#.reset_index(drop=True)
        tst_ = tst_.sort_index()#.reset_index(drop=True)

#        leg_ = pd.concat([trn_.iloc[:,-2], tst_.iloc[:,-2]], ignore_index=False)
        leg_ = pd.concat([trn_, tst_], ignore_index=False)

        # %matplotlib qt
        fig, ax = plt.subplots(1, 1, sharex=False, tight_layout=True, figsize=(12,6))
        ax.scatter(leg_.index, leg_, color='green', s=10, marker='o')
        ax.scatter(trn_.index, trn_, color='red', s=10, marker='x')
        ax.scatter(tst_.index, tst_, color='blue', s=15, marker='+') # [y_h]
    #    ax.plot(leg_whole_ts.index, leg_whole_ts.iloc[:,-1], color='green', linewidth=1)
        ax.legend(['y ground truth', 'train prediction', 'test prediction'])

    del trn_, tst_, leg_

##############################################################################################################
def aggregated_bins_regression_plot_weights(stats,sets,options,colors,SAVE=True):
  # aggregated_bins_regression_plot_weights
    # Produce plots of weight importance, given set of weight parameters
    try:
        len_plot = len(options['LEG_P'])
    except TypeError:
        len_plot = 1
        options['LEG_P'] = [options['LEG_P']]

    bar_w = 0.1
    # print(savefigs)

    for sea in options['SEA']:
        for sep_method in options['SEP_METHOD']:
            for varset in options['VARSET']:

                try:
                    tickname = dataset.subset_data_stack_variables([],
                        varset, seatype=sea, mode='returnnames')
                except AttributeError:
                    tickname = [str(aa) for aa in range(options['DATA_DIM'])]

                fig, ax = plt.subplots(len(sets), len_plot, sharey=False, tight_layout=False,
                                       figsize=(5*len_plot,7), squeeze=False)

                for indbi, bin_ in enumerate(sets):
                    index = np.arange(len(tickname))
                    for ind, meth in enumerate(options['REGR_WITH_WEIGTHS']):
                        for legind,leg in enumerate(options['LEG_P']):
                            string_plots = sea + '_leg_' + str(leg) + '_' + sep_method + '_' + \
                            meth + '_' +  varset

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

                    index = np.arange(len(tickname))

                    ax[indbi,0].set_ylabel('' + bin_)

                # if indbi < len(sets)-1:
                    # ax[indbi,leg-1].set_xticklabels([])

                # if indbi == len(sets)-1:
                    # if len(tickname) > 6:
                    # ax[-1,leg-1].set_xticklabels(tickname, rotation=90)
                    # else:
                    # ax[-1,leg-1].set_xticklabels(tickname)

                index = np.arange(len(tickname))


                plt.legend(options['REGR_WITH_WEIGTHS'])
                plt.suptitle(sea + '_leg_' + str(leg) + '_' + sep_method + '_' +  varset)
                plt.show(block=False)#

                if SAVE:
                    plt.savefig(options['SAVEFOLDER'] + 'weights_' + 'agg_bins' + '_' + sea + '_leg_' + str(leg) + '_' + \
                         sep_method + '_' +  varset + '.png', bbox_inches='tight')

##############################################################################################################
def aggregated_bins_regression_plot_errors(stats,sets,options,colors,SAVE=True):

    try:
        len_plot = len([options['LEG_P']])
    except TypeError:
        len_plot = 1
        options['LEG_P'] = [options['LEG_P']]

    bar_w = 0.05
    for errmeasure in options['ERRMEASURE']:
        for sea in options['SEA']:
            for sep_method in options['SEP_METHOD']:
                for varset in options['VARSET']:
                    fig, ax = plt.subplots(2, len_plot, sharey=False,
                                           tight_layout=False, figsize=(5*len_plot,7), squeeze=False)

                    for legind, leg in enumerate([options['LEG_P']]):
                        index = np.arange(1)
                        for indbi, bin_ in enumerate(sets):
                            for ind, meth in enumerate(options['METHODS']):
                                string_plots = sea + '_leg_' + str(leg) + '_' + sep_method + '_' + \
                                    meth + '_' +  varset

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

                            if leg == 1:
                                ax[0,0].set_ylabel('training ' + errmeasure.upper())
                                ax[1,0].set_ylabel('testing ' + errmeasure.upper())

                            ax[1,legind].set_xlabel('LEG ' + str(leg), fontsize=16)

    #                             ax[1,leg-1].set_xticks(index + len(options['METHODS'])*bar_w/2 - bar_w/2)
                            loc = [3/2*bar_w + len(options['METHODS'])*bar_w/2 + ll for ll in (2*bar_w+bar_w*len(options['METHODS']))*np.arange(0,len(sets),1)]


                            if errmeasure.lower() == 'r2':
                                ax[0,legind].set_ylim([-0.5,1])
                                ax[0,legind].set_yticks(np.arange(-0.5,1,0.1))
                            ax[0,legind].set_xticks(np.arange(len(sets))/len(sets)+bar_w/2)
                            ax[0,legind].set_xticklabels('')
                            ax[0,legind].grid(axis='y')
                            ax[0,legind].grid(color='black', which='both', axis='y', linestyle=':')

                            if errmeasure.lower() == 'r2':
                                ax[1,legind].set_ylim([-0.5,1])
                                ax[1,legind].set_yticks(np.arange(-0.5,1,0.1))
                            ax[1,legind].set_xticks(np.arange(len(sets))/len(sets)+bar_w/2)
                            ax[1,legind].set_xticklabels(sets)
                            ax[1,legind].grid(axis='y')
                            ax[1,legind].grid(color='black', which='both', axis='y', linestyle=':')


                    plt.legend(options['METHODS'])
                    plt.suptitle(sea + '_leg_' + str(leg) + '_' + sep_method + '_' +  varset)
                    plt.show(block=False)#

                    if SAVE:
                        plt.savefig(options['SAVEFOLDER'] + errmeasure + '_' + sea + '_leg_' + str(leg) + '_' + \
                                    sep_method + '_' +  varset + '.png', bbox_inches='tight')



##############################################################################################################
def single_bins_regression_plot_weights(stats,sets,options,colors,SAVE=True, ylim=[None, None]):

    try:
        len_plot = len([options['LEG_P']])
    except TypeError:
        len_plot = 1
        options['LEG_P'] = [options['LEG_P']]

    bar_w = 0.75
    for sea in options['SEA']:
        for sep_method in options['SEP_METHOD']:
            for varset in options['VARSET']:

                tickname = options['VARNAMES']
# dataset.subset_data_stack_variables([],
#         varset, seatype=sea, mode='returnnames')

                for ind, meth in enumerate(options['REGR_WITH_WEIGTHS']):
                    index = np.arange(len(options['COLNAMES']))
                    fig, ax = plt.subplots(len(tickname), len_plot, sharey=False,
                                    tight_layout=False,
                                    figsize=(15,10), squeeze=False)

                    for legind, leg in enumerate(options['LEG_P']):
                        for ind_w, parname in enumerate(tickname):
                            string_plots = sea + '_leg_' + str(leg) + '_' + sep_method + '_' + \
                            meth + '_' +  varset
                            w = []
                            s = []

                            for bin_ in sets:
                                w.append(np.squeeze(stats[bin_][string_plots]['weights'][0][ind_w]))#[0][ind_w])#[string_plots + '_mean']
                                s.append(np.squeeze(stats[bin_][string_plots]['weights'][1][ind_w]))#[ind_w] [string_plots + '_mean']

                            w = np.array(w)
                            s = np.array(s)

                            # w[w > 10] = 10
                            # print(ind_w, legind, w.shape, s.shape, )
                            ax[ind_w,legind].bar(index, w, bar_w,
                                                 color=tuple(colors[ind,:]), yerr=s)#olors[ind]'
                            index = index + bar_w

                            if leg == 1:
                                ax[ind_w,legind].set_ylabel(parname)

                            index = np.arange(0,len(sets),1)
                            ax[ind_w,legind].set_xticks(index)
                            ax[ind_w,legind].set_xticklabels('')
        #                     ax[ind_w-1,leg-1].set_yticks(np.arange(-2,2,0.5))
        #                     ax[ind_w-1,leg-1].set_yticklabels(np.arange(-2,2,0.5),minor=False)
                            ax[ind_w,legind].grid(color='black', which='both', axis='y', linestyle=':')
                            ax[ind_w,legind].set_ylim(ylim)

                            for c in options['AGGREGATES']:
                                ax[ind_w,legind].axvline(c)

    #             plt.legend([meth + ' ' + sep_method + ', param: ' + parname],loc=0)
                    plt.suptitle(meth + '_' + sea + '_leg_' + str(leg) + '_' + sep_method + '_' +  varset)
                    ax[-1,-1].set_xlabel('LEG ' + str(leg), fontsize=16)
                    ax[-1,-1].set_xticklabels(options['COLNAMES'],fontsize=5,rotation='vertical')
                    plt.show(block=False)#

                    if SAVE:
                        plt.savefig(options['SAVEFOLDER'] + 'weights_' + meth + '_' + sea + '_leg_' + str(leg) + '_' + \
                                        sep_method + '_' +  varset + '.png', bbox_inches='tight')


##############################################################################################################
def single_bins_regression_plot_errors(stats,sets,options,colors,SAVE=True):

    try:
        len_plot = len([options['LEG_P']])
    except TypeError:
        len_plot = 1
        options['LEG_P'] = [options['LEG_P']]

    bar_w = 0.45
    index = np.arange(len(options['ERRMEASURE']))
    for errmeasure in options['ERRMEASURE']:
        for sea in options['SEA']:
            for sep_method in options['SEP_METHOD']:
                for varset in options['VARSET']:

                    # tickname = dataset.subset_data_stack_variables([],
                            # varset, seatype=sea, mode='returnnames')
                    index = np.arange(colors.shape[0])

                    for ind, meth in enumerate(options['METHODS']):

                        index = np.arange(len(sets))
                        fig, [ax] = plt.subplots(1, len_plot, sharey=False, tight_layout=False,
                        figsize=(15,5), squeeze=False)

                        for legind,leg in enumerate(options['LEG_P']):
                            string_plots = sea + '_leg_' + str(leg) + '_' + sep_method + '_' + \
                            meth + '_' +  varset
                            e_tr = []; e_ts = []
                            s_tr = []; s_ts = []

                            for bin_ in sets:
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

    #                         l1 = ax[leg-1].bar(index, e_ts, bar_w, color=tuple(colors[0,:]), yerr=s_ts)
    #                         l2 = ax[leg-1].bar(index+bar_w, e_ts, bar_w, color=tuple(colors[1,:]), yerr=s_ts)

                            if leg == 1:
                                ax[legind].set_ylabel(errmeasure)
                                ax[legind].legend()

                            ax[legind].set_xticks(index)
                            ax[legind].set_xticklabels('')

                            if errmeasure.lower() == 'r2':
                                ax[legind].set_ylim([-0.5,1])
                            ax[legind].grid(color='black', which='both', axis='y', linestyle=':')

                        for c in options['AGGREGATES']:
                            ax[legind].axvline(c)

                        plt.suptitle(errmeasure + '_' + meth + '_' + sea + '_leg_' + str(leg) + '_' + sep_method + '_' +  varset)
                        ax[-1].set_xlabel('LEG ' + str(leg), fontsize=16)
                        ax[-1].set_xticklabels(options['COLNAMES'],fontsize=5,rotation='vertical')
                        plt.show(block=False)#

                        if SAVE:
                            plt.savefig(options['SAVEFOLDER'] + errmeasure + '_' + meth + '_' + sea + '_leg_' + str(leg) + '_' + \
                                sep_method + '_' +  varset + '.png', bbox_inches='tight')

##############################################################################################################
def visualize_stereo_map_DEPRECATED(coordinates, values, min_va, max_va, markersize=75, fillconts='grey', fillsea='aqua', labplot='', plottype='scatter'):
    '''
        Visualize data on a polar stereographic projection map using Basemap on matplotlib. It probably needs to be updated in the future as this package is no longer mantained since easily 2013-2014 or something like that. But I don't know about options that are as easy and as flexible (it is basically matplotlib)

            INPUTS
                - coordinates: those are basically fixed, from the boad path. Still given as argument for flexibility
                - values: can be a 1D vector passing values (and a colormap is build accordingly) or it can be a 2D vector Nx3 corresponding to some colormap to be used as plots
                - fill* : colors to fill continents and seas.
                - label : the label for the scatter series, to use for legend and so on
                - min_, max_: min / max values to clip variable to plot. Default are min / max of the series
                - plottype : (BETA) use differnt plotting tools (e.g. scatter or plot, etc.)
            OUTPUTS
                - the most awesome map of the antarctic continent, without penguins

            EXAMPLE

            TODO
                - Add support for _custom_ background image (e.g. sea surface temperature, wind magnitude, etc.) (use Basemap.contourf() to interpolate linearly within a grid of values)

            NOTE
                - The longitude lon_0 is at 6-o'clock, and the latitude circle boundinglat is tangent to the edge of the map at lon_0. Default value of lat_ts (latitude of true scale) is pole.
                - Latitude is in °N, longitude in is °E
    '''

    if coordinates.shape[0] != values.shape[0]:
        print('size of gps coordinates and variable to be plotted does not match')
        return

    # prepare basemap
    m = carto.Basemap(projection='spstere',boundinglat=-32,lon_0=180,resolution='l')

    m.drawcoastlines()
    m.fillcontinents(color=fillconts)
    m.drawmapboundary(fill_color=fillsea)
    m.drawparallels(np.arange(-90.,81.,20.))
    m.drawmeridians(np.arange(-180.,181.,20.))
    #m.shadedrelief()
    # prepare colors
    cmap = plt.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=min_va, vmax=max_va)
    colors = [cmap(normalize(value)) for value in values]

    # geo coord to plot coord
    lon, lat = m(coordinates.iloc[:,1].values,coordinates.iloc[:,0].values)

    # map boat samples with values
    if plottype == 'scatter':
        im = m.scatter(lon,lat,color=colors,s=markersize, linewidth=0, label=labplot)
    elif plottype == 'plot':
        im = m.plot(lon,lat,color=colors,linewidth=markersize,label=labplot)
    else:
        print('unrecognized plot')
        return

    ax = plt.gca()
    # ax.set_title(labplot,fontsize=35)

    cax, _ = clb.make_axes(ax)
    cbar = clb.ColorbarBase(cax, cmap=cmap, norm=normalize)

    return im, ax

##############################################################################################################
def visualize_stereo_map(coordinates, values, min_va, max_va, markersize=75, fillconts='grey', fillsea='aqua', labplot='', plottype='scatter'):
    '''
        Visualize data on a polar stereographic projection map using Basemap on matplotlib. It probably needs to be updated in the future as this package is no longer mantained since easily 2013-2014 or something like that. But I don't know about options that are as easy and as flexible (it is basically matplotlib)

            INPUTS
                - coordinates: those are basically fixed, from the boad path. Still given as argument for flexibility
                - values: can be a 1D vector passing values (and a colormap is build accordingly) or it can be a 2D vector Nx3 corresponding to some colormap to be used as plots
                - fill* : colors to fill continents and seas.
                - label : the label for the scatter series, to use for legend and so on
                - min_, max_: min / max values to clip variable to plot. Default are min / max of the series
                - plottype : (BETA) use differnt plotting tools (e.g. scatter or plot, etc.)
            OUTPUTS
                - the most awesome map of the antarctic continent, without penguins

            EXAMPLE

            TODO
                - Add support for _custom_ background image (e.g. sea surface temperature, wind magnitude, etc.) (use Basemap.contourf() to interpolate linearly within a grid of values)

            NOTE
                - The longitude lon_0 is at 6-o'clock, and the latitude circle boundinglat is tangent to the edge of the map at lon_0. Default value of lat_ts (latitude of true scale) is pole.
                - Latitude is in °N, longitude in is °E
    '''

    if coordinates.shape[0] != values.shape[0]:
        print('size of gps coordinates and variable to be plotted does not match')
        return

    fig = plt.gcf()
    # prepare basemap
    ax = fig.add_subplot(111, projection=ccrs.SouthPolarStereo())

    # ax.set_proj
    ax.set_extent([-180, 180, -90, -35], ccrs.PlateCarree())
    ax.coastlines(linewidth=1.5)
    ax.add_feature(cfeature.LAND, facecolor=fillconts)
    ax.add_feature(cfeature.OCEAN, facecolor=fillsea)
    ax.gridlines(color='black', linestyle='--', alpha=0.5)
    #m.shadedrelief()
    # prepare colors
    cmap = plt.cm.get_cmap('viridis')
    normalize = mpl.colors.Normalize(vmin=min_va, vmax=max_va)
    colors = [cmap(normalize(value)) for value in values]

    # geo coord to plot coord
    # lon, lat = m(coordinates.iloc[:,1].values,coordinates.iloc[:,0].values)
    # map boat samples with values
    if plottype == 'scatter':
        ax.scatter(coordinates.iloc[:,1].values,coordinates.iloc[:,0].values,
            transform=ccrs.PlateCarree(),
            color=colors,s=markersize, linewidth=0, label=labplot)
    elif plottype == 'plot':
        ax.plot(coordinates.iloc[:,1].values,coordinates.iloc[:,0].values,
            transform=ccrs.PlateCarree(),
            color=colors,s=markersize, linewidth=0, label=labplot)
        # im = m.plot(lon,lat,color=colors,linewidth=markersize,label=labplot)
    else:
        print('unrecognized plot')
        return

    # ax.set_title(labplot,fontsize=35)

    cax, _ = clb.make_axes(ax)
    cbar = clb.ColorbarBase(cax, cmap=cmap, norm=normalize)

    return ax

##############################################################################################################
def scatterplot_matrix(df, color=None):

    df.columns = [str(cc) for cc in df.columns]

    nrows, ncols = df.shape[1], df.shape[1]
    fig, ax = plt.subplots(nrows, ncols, sharex=False, sharey=False, tight_layout=True, figsize=(10,10))

    row = 0; col = 0;
    for row, r_name in enumerate(df.columns):
        for col, c_name in enumerate(df.columns):
            if col == row:
                ax[row,col].hist(df.iloc[:,row],bins=100, histtype='step')
            elif col != row:
                ax[row,col].scatter(df.iloc[:,row],df.iloc[:,col],c=color)

            if col == 0:
                ax[row,col].set_ylabel(r_name)
            else:
                ax[row,col].set_ylabel('')

            if row == nrows-1:
                ax[row,col].set_xlabel(c_name)
            else:
                ax[row,col].set_xlabel('')

    return fig, ax
