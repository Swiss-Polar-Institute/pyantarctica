import matplotlib.pyplot as plt


def plot_predicted_timeseries(leg_whole, trn, tst, y_ts_h, y_tr_h, y_gt_scaled, SEP_METHOD):
    # Should probably be moved in a visualization module, here not really modeling stuff... 

    if SEP_METHOD is 'time':
        # %matplotlib qt
        trn_ = trn.reset_index(drop=True)
        s1 = len(trn_)
        tst_ = tst.reset_index(drop=True)
        tst_.index = tst_.index +  s1
        leg_whole_ = leg_whole.reset_index(drop=True)
        
        fig, ax = plt.subplots(1, 1, sharex=False, tight_layout=True, figsize=(12,6))
        ax.plot(trn_.index, y_tr_h, color='red', linewidth=2)
        ax.plot(tst_.index, y_ts_h, color='blue', linewidth=2)
    #    ax.plot(leg_whole_ts.index, leg_whole_ts.iloc[:,-1], color='green', linewidth=1)
        ax.plot(leg_whole_.index, y_gt_scaled, color='green', linewidth=1)
        ax.legend(['train prediction', 'test prediction', 'y ground truth' ])
    
    elif SEP_METHOD is 'random':
        
        trn_ = trn.copy()
        tst_ = tst.copy()
        leg_whole_ = leg_whole.copy()
        trn_['y_h'] = y_tr_h
        tst_['y_h'] = y_ts_h
        leg_whole_['y_gt_scaled'] = y_gt_scaled
        trn_ = trn_.sort_index()
        tst_ = tst_.sort_index()
        leg_whole_ = leg_whole_.sort_index()

        # %matplotlib qt
        fig, ax = plt.subplots(1, 1, sharex=False, tight_layout=True, figsize=(12,6))
        ax.scatter(leg_whole_.index, leg_whole_['y_gt_scaled'], color='green', s=10, marker='o')
        ax.scatter(trn_.index, trn_['y_h'], color='red', s=10, marker='x')
        ax.scatter(tst_.index, tst_['y_h'], color='blue', s=15, marker='+')
    #    ax.plot(leg_whole_ts.index, leg_whole_ts.iloc[:,-1], color='green', linewidth=1)
        ax.legend(['y ground truth', 'train prediction', 'test prediction'])

    del trn_, tst_, leg_whole_
