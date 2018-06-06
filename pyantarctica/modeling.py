import pandas as pd
import numpy as np

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


def sample_trn_test_index(index,split=2.0/3,N_tst=20,mode='final'):
        
    if mode == 'final':
        s0 = len(index)
        ind_set = np.ones((s0,1))

        s1 = int(np.floor(s0*split))
        s2 = s0-s1
        #print(s2)
        
        if N_tst == 'all': 
            ind_set[(s1+1):] = 2
            return
        else:            
            for bl, ii in enumerate(range(0,s2,N_tst)):
                #     print(bl+2,s0+1+ii,s0+1+ii+N_tst)
                    ind_set[(s1+1+ii):(s1+ii+1+N_tst)] = bl+2

    if mode == 'middle':
        s0 = len(index)
        ind_set = np.ones((s0,1))

        s1 = int(np.floor(s0*split))
        s2 = len(ind_set)-s1
        
        lab = 2
        for bl, ii in enumerate(range(0,s1+s2,N_tst)):
            #print(int((bl%(split))*100))
            if int((bl%(split))*100) == 0:
                ind_set[(1+ii):(ii+1+N_tst)] = 1
            else: 
                ind_set[(1+ii):(ii+1+N_tst)] = lab
                lab += 1

        # ind_set[(s0+ii+N_tst):(s0+ii+s1-ii-1)] = 0#bl+3
        
    if mode=='initial':
        sample_trn_test_index(np.ones((len(index),1)),split=2.0/3,N_tst=20,mode='final')
        ind_set = ind_set[-1::-1]
            
    return pd.DataFrame(ind_set.reshape(-1,1), index=index, columns=['ind'])
