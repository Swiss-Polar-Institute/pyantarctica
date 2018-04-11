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


def retrieve_model_av_std(summary):
    exps_ = [s[:-2] for s in list(summary.keys())]
    exps = set(exps_)
    NUM_REP = int(len(exps_) / len(exps))
#    print(NUM_REP)
#    print(set(exps))
    results = {}
    for name_ in exps: 
    #     print(name_)

        results[name_] = {}

        init_ = True
        for nre in range(0,NUM_REP,1):
            sub_res = summary[name_ + '_' + str(nre)]
            if init_:
                for sub_, val_ in sub_res.items():  
                        exec(sub_ + '= []' )
                        init_ = False

            for sub_, val_ in sub_res.items():  
    #             print('-> ', sub_,val_)
                exec(sub_+'.append(val_)')

        for sub_ in sub_res:
            if '_hat' not in sub_:
                exec(sub_ + '= np.array(' + sub_ + ')')
    #         print(eval(sub_))

        for sub_ in sub_res:
            if '_hat' not in sub_:
                exec('results[name_][sub_] = np.append(np.mean([' + sub_ + '], axis=1), np.std([' + sub_ + '], axis=1),axis=0)')  
            else:
                exec('results[name_][sub_] =' + sub_)

    return results
