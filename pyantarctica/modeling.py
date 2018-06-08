import numpy as np
import pandas as pd


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


def sample_trn_test_index(index,split=2.0/3,group=20,mode='final'):
    """
    SAMPLE TRN / TST INDEXES IN SOME PARTICULAR ORDER:
         index : pd.index -- dataframe index of the dataset from which sample from (required to return a dataframe as well, without losing the original indexing)
         split: float -- training to test datapoints ratioself.
         group: int or 'all' --- number of samples in each test subgroup, if needed, and for other things.
         mode: string --
                - final : first split% used for training, rest for testing
                - middle : samples equal groups for training and groups for testing, with size specified by group, independent training are all indexed by 1 and the tests are independent groups with label l in {2,...}, uniformly distributed
                - initial : recursively uses 'final' but on inverted indexes
                - training_shifted : indexes training points in temporally shifted lags, with group specifying how many points _before_ the training set is sampled.
    """

    s0 = len(index)
    ind_set = np.ones((s0,1))

    s1 = int(np.floor(s0*split))
    s2 = s0-s1

    if mode == 'final':
        if N_tst == 'all':
            ind_set[(s1+1):] = 2
            return
        else:
            for bl, ii in enumerate(range(0,s2,group)):
                ind_set[(s1+1+ii):(s1+ii+1+group)] = bl+2

    elif mode == 'middle':
        lab = 2
        for bl, ii in enumerate(range(0,s1+s2,group)):
            print(int((bl%(split))*100))
            if int((bl%(split))) == 0:
                ind_set[(1+ii):(ii+1+group)] = 1
            else:
                ind_set[(1+ii):(ii+1+group)] = lab
                lab += 1

    elif mode=='initial':
        ind_set = sample_trn_test_index(np.ones((len(index),1)),split=2.0/3,N_tst=20,mode='final')
        ind_set = ind_set[-1::-1]

    elif mode=='training_shifted':

        ind_set[0:group] = 2
        ind_set[(group+s1):] = 3

    return pd.DataFrame(ind_set.reshape(-1,1), index=index, columns=['ind'])
