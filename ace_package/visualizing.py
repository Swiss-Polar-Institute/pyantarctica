import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
                    for ind, meth in enumerate(options['METHODS']):
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


def plot_prediction_on_aggregations(stats,sets,options,colors,SAVE=True):
    return