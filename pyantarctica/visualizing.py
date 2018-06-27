import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import mpl_toolkits.basemap as carto


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

def aggregated_bins_regression_plot_weights(stats,sets,options,colors,SAVE=True):
  # aggregated_bins_regression_plot_weights
    # Produce plots of weight importance, given set of weight parameters

    bar_w = 0.1
    # print(savefigs)

    for sea in options['SEA']:
        for sep_method in options['SEP_METHOD']:
            for varset in options['VARSET']:
                if varset.lower() == 'full':
                    Nw=6
                    if sea == 'total':
                        tickname = ['hs', 'tp', 'steep', 'phase_vel', 'age', 'wind']
                    elif sea == 'wind':
                        tickname  = ['hs_w', 'tp_w', 'steep_w', 'phase_vel_w', 'age_w', 'wind']

                elif varset.lower() == 'nowind':
                    Nw=4
                    if sea == 'total':
                        tickname = ['hs', 'tp', 'steep', 'phase_vel']
                    elif sea == 'wind':
                        tickname  = ['hs_w', 'tp_w', 'steep_w', 'phase_vel_w']

                elif varset.lower() == 'reduced':
                    Nw=3
                    if sea == 'total':
                        tickname = ['hs', 'tp', 'wind']
                    elif sea == 'wind':
                        tickname  = ['hs_w', 'tp_w', 'wind']

                index = np.arange(Nw)

                fig, ax = plt.subplots(len(sets), len(options['LEG_P']), sharey=False, tight_layout=False,
                                       figsize=(10,10*len(options['LEG_P'])), squeeze=False)

                for indbi, bin_ in enumerate(sets):
                    for ind, meth in enumerate(options['REGR_WITH_WEIGTHS']):
                        for leg in options['LEG_P']:
                            string_plots = sea + '_leg_' + str(leg) + '_' + sep_method + '_' + \
                            meth + '_' +  varset

                            if 'weights' in  stats[bin_][string_plots]:
                                if meth == 'rbfgprard':
                                    w = 1/(1 + 10*stats[bin_][string_plots]['weights'][0])#[string_plots + '_mean']
                                    s = 1/(1 + 10*stats[bin_][string_plots]['weights'][1])#[string_plots + '_mean']
                                else:
                                    w = stats[bin_][string_plots]['weights'][0]#[string_plots + '_mean']
                                    s = stats[bin_][string_plots]['weights'][1]#[string_plots + '_mean']
                            else:
                                break

                            if bin_ == '80-200':
                                print(w)

                            ax[indbi,leg-1].bar(index, w, bar_w, color=tuple(colors[ind,:]))#, yerr=s)#olors[ind]'

                            index = index + bar_w

                    if leg == 1:
                        ax[indbi,leg-1].set_ylabel('Scores ' + bin_)

                    ax[-1,leg-1].set_xlabel('LEG ' + str(leg), fontsize=16)

                    index = np.arange(Nw)
                    ax[indbi,leg-1].set_xticks(index + bar_w)
                    ax[indbi,leg-1].set_xticklabels(tickname)

                plt.legend(options['METHODS'])
                plt.suptitle(sea + '_leg_' + str(leg) + '_' + sep_method + '_' +  varset)
                plt.show()

                if SAVE:
                    plt.savefig(options['SAVEFOLDER'] + 'weights_' + 'agg_bins' + '_' + sea + '_leg_' + str(leg) + '_' + \
                         sep_method + '_' +  varset + '.png', bbox_inches='tight')


def aggregated_bins_regression_plot_errors(stats,sets,options,colors,SAVE=True):

    bar_w = 0.05
    for errmeasure in options['ERRMEASURE']:
        for sea in options['SEA']:
            for sep_method in options['SEP_METHOD']:
                for varset in options['VARSET']:
                    index = np.arange(1)
                    fig, ax = plt.subplots(2, len(options['LEG_P']), sharey=False,
                                           tight_layout=False, figsize=(10*len(options['LEG_P']),10), squeeze=False)

                    for indbi, bin_ in enumerate(sets):
                        index = index + 2*bar_w
                        for ind, meth in enumerate(options['METHODS']):
                            for leg in options['LEG_P']:
                                string_plots = sea + '_leg_' + str(leg) + '_' + sep_method + '_' + \
                                    meth + '_' +  varset

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

        #                             print(string_plots)
    #                             print('tst:' + str(e_ts))
    #                             print('trn:' + str(e_tr))

                                ax[0, leg-1].bar(index, e_tr, bar_w, color=tuple(colors[ind,:]), yerr=s_tr)
                                ax[1, leg-1].bar(index, e_ts, bar_w, color=tuple(colors[ind,:]), yerr=s_ts)

                                index = index + bar_w
                                if leg == 1:
                                    ax[0,0].set_ylabel('training ' + errmeasure.upper())
                                    ax[1,0].set_ylabel('testing ' + errmeasure.upper())

                                ax[1,leg-1].set_xlabel('LEG ' + str(leg), fontsize=16)

        #                             ax[1,leg-1].set_xticks(index + len(options['METHODS'])*bar_w/2 - bar_w/2)
                                loc = [3/2*bar_w + len(options['METHODS'])*bar_w/2 + ll for ll in (2*bar_w+bar_w*len(options['METHODS']))*np.arange(0,len(sets),1)]


                                if errmeasure.lower() == 'r2':
                                    ax[0,leg-1].set_ylim([-0.5,1])
                                    ax[0,leg-1].set_yticks(np.arange(-0.5,1,0.1))
                                ax[0,leg-1].set_xticks(loc)
                                ax[0,leg-1].set_xticklabels('')
                                ax[0,leg-1].grid(axis='y')
                                ax[0,leg-1].grid(color='black', which='both', axis='y', linestyle=':')

                                if errmeasure.lower() == 'r2':
                                    ax[1,leg-1].set_ylim([-0.5,1])
                                    ax[1,leg-1].set_yticks(np.arange(-0.5,1,0.1))
                                ax[1,leg-1].set_xticks(loc)
                                ax[1,leg-1].set_xticklabels(sets)
                                ax[1,leg-1].grid(axis='y')
                                ax[1,leg-1].grid(color='black', which='both', axis='y', linestyle=':')


                    plt.legend(options['METHODS'])
                    plt.suptitle(sea + '_leg_' + str(leg) + '_' + sep_method + '_' +  varset)
                    plt.show()

                    if SAVE:
                        plt.savefig(options['SAVEFOLDER'] + errmeasure + '_' + sea + '_leg_' + str(leg) + '_' + \
                                    sep_method + '_' +  varset + '.png', bbox_inches='tight')



def single_bins_regression_plot_weights(stats,sets,options,colors,SAVE=True):

    bar_w = 0.75
    for sea in options['SEA']:
        for sep_method in options['SEP_METHOD']:
            for varset in options['VARSET']:
                if varset.lower() == 'full':
                    Nw=6
                    if sea == 'total':
                        tickname = ['hs', 'tp', 'steep', 'phase_vel', 'age', 'wind']
                    elif sea == 'wind':
                        tickname  = ['hs_w', 'tp_w', 'steep_w', 'phase_vel_w', 'age_w', 'wind']

                elif varset.lower() == 'nowind':
                    Nw=4
                    if sea == 'total':
                        tickname = ['hs', 'tp', 'steep', 'phase_vel']
                    elif sea == 'wind':
                        tickname  = ['hs_w', 'tp_w', 'steep_w', 'phase_vel_w']

                elif varset.lower() == 'reduced':
                    Nw=3
                    if sea == 'total':
                        tickname = ['hs', 'tp', 'wind']
                    elif sea == 'wind':
                        tickname  = ['hs_w', 'tp_w', 'wind']

                index = np.arange(len(sets))


                for ind, meth in enumerate(options['REGR_WITH_WEIGTHS']):
                    fig, ax = plt.subplots(len(tickname), len(options['LEG_P']), sharey=False, tight_layout=False,
                                figsize=(15,10), squeeze=False)

                    for leg in options['LEG_P']:
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

                            w[w > 10] = 10
                            ax[ind_w, leg-1].bar(index, w, bar_w,
                                                 color=tuple(colors[ind,:]), yerr=s)#olors[ind]'
                            index = index + bar_w

                            if leg == 1:
                                ax[ind_w, leg-1].set_ylabel(parname)

                            index = np.arange(0,len(sets),1)
                            ax[ind_w,leg-1].set_xticks(index)
                            ax[ind_w,leg-1].set_xticklabels('')
        #                     ax[ind_w-1,leg-1].set_yticks(np.arange(-2,2,0.5))
        #                     ax[ind_w-1,leg-1].set_yticklabels(np.arange(-2,2,0.5),minor=False)
                            ax[ind_w,leg-1].grid(color='black', which='both', axis='y', linestyle=':')

                            for c in options['AGGREGATES']:
                                ax[ind_w,leg-1].axvline(c)

    #             plt.legend([meth + ' ' + sep_method + ', param: ' + parname],loc=0)
                    plt.suptitle(meth + '_' + sea + '_leg_' + str(leg) + '_' + sep_method + '_' +  varset)
                    ax[-1,-1].set_xlabel('LEG ' + str(leg), fontsize=16)
                    ax[-1,-1].set_xticklabels(options['COLNAMES'],fontsize=5,rotation='vertical')
                    plt.show()

                    if SAVE:
                        plt.savefig(options['SAVEFOLDER'] + 'weights_' + meth + '_' + sea + '_leg_' + str(leg) + '_' + \
                                        sep_method + '_' +  varset + '.png', bbox_inches='tight')



def single_bins_regression_plot_errors(stats,sets,options,colors,SAVE=True):

    bar_w = 0.45
    index = np.arange(len(options['ERRMEASURE']))
    for errmeasure in options['ERRMEASURE']:
        for sea in options['SEA']:
            for sep_method in options['SEP_METHOD']:
                for varset in options['VARSET']:
                    if varset.lower() == 'full':
                        Nw=6
                        if sea == 'total':
                            tickname = ['hs', 'tp', 'steep', 'phase_vel', 'age', 'wind']
                        elif sea == 'wind':
                            tickname  = ['hs_w', 'tp_w', 'steep_w', 'phase_vel_w', 'age_w', 'wind']

                    elif varset.lower() == 'nowind':
                        Nw=4
                        if sea == 'total':
                            tickname = ['hs', 'tp', 'steep', 'phase_vel']
                        elif sea == 'wind':
                            tickname  = ['hs_w', 'tp_w', 'steep_w', 'phase_vel_w']

                    elif varset.lower() == 'reduced':
                        Nw=3
                        if sea == 'total':
                            tickname = ['hs', 'tp', 'wind']
                        elif sea == 'wind':
                            tickname  = ['hs_w', 'tp_w', 'wind']


                    for ind, meth in enumerate(options['METHODS']):

                        index = np.arange(len(sets))
                        fig, [ax] = plt.subplots(1, len(options['LEG_P']), sharey=False, tight_layout=False,
                        figsize=(15,5), squeeze=False)

                        for leg in options['LEG_P']:
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


                            l1 = ax[leg-1].plot(e_tr, color=tuple(colors[0,:]),label='train')#, yerr=s_tr)
                            l2 = ax[leg-1].plot(e_ts, color=tuple(colors[1,:]),label='test')#, yerr=s_tr) index+bar_w

    #                         l1 = ax[leg-1].bar(index, e_ts, bar_w, color=tuple(colors[0,:]), yerr=s_ts)
    #                         l2 = ax[leg-1].bar(index+bar_w, e_ts, bar_w, color=tuple(colors[1,:]), yerr=s_ts)

                            if leg == 1:
                                ax[leg-1].set_ylabel(errmeasure)
                                ax[leg-1].legend()

                            ax[leg-1].set_xticks(index)
                            ax[leg-1].set_xticklabels('')

                            if errmeasure.lower() == 'r2':
                                ax[leg-1].set_ylim([-0.5,1])
                            ax[leg-1].grid(color='black', which='both', axis='y', linestyle=':')

                        for c in options['AGGREGATES']:
                            ax[leg-1].axvline(c)

                        plt.suptitle(errmeasure + '_' + meth + '_' + sea + '_leg_' + str(leg) + '_' + sep_method + '_' +  varset)
                        ax[-1].set_xlabel('LEG ' + str(leg), fontsize=16)
                        ax[-1].set_xticklabels(options['COLNAMES'],fontsize=5,rotation='vertical')
                        plt.show()

                        if SAVE:
                            plt.savefig(options['SAVEFOLDER'] + errmeasure + '_' + meth + '_' + sea + '_leg_' + str(leg) + '_' + \
                                sep_method + '_' +  varset + '.png', bbox_inches='tight')

def visualize_stereo_map(coordinates, values, markersize=0.75, fillconts='grey', fillsea='aqua'):
    '''
        Visualize data on a polar stereographic projection map using Basemap on matplotlib. It probably needs to be updated in the future as this package is no longer mantained since easily 2013-2014 or something like that. But I don't know about options that are as easy and as flexible (it is basically matplotlib)
            INPUTS
                - coordinates: those are basically fixed, from the boad path. Still given as argument for flexibility
                - values: can be a 1D vector passing values (and a colormap is build accordingly) or it can be a 2D vector Nx3 corresponding to some colormap to be used as plots
                - fill* : colors to fill continents and seas.

            OUTPUTS
                - the most awesome map of the antarctic continent, without penguins

            EXAMPLE

            TODO
                - Add support for background color (e.g. sea surface temperature, wind magnitude, etc.)


    '''

    m = carto.Basemap(projection='spstere',boundinglat=-32,lon_0=180,resolution='l')
    # m = carto.Basemap(projection='stere',lat_0=-90,lon_0=0,width=14000000, height=14000000)

    m.drawcoastlines()
    m.fillcontinents(color=fillconts)
    m.drawmapboundary(fill_color=fillsea)

    m.drawparallels(np.arange(-90.,81.,20.))
    m.drawmeridians(np.arange(-180.,181.,20.))

    # draw parallels and meridians.
    # draw tissot's indicatrix to show distortion.
    lon, lat = m(coordinates.iloc[:,1].values,coordinates.iloc[:,0].values)

    m.scatter(lon,lat,linewidth=0.4,color=values)
    plt.show()
