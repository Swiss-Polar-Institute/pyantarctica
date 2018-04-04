import matplotlib.pyplot as plt
import pandas as pd

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
    
    
    
    
    
    return None, None
    