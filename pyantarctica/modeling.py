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

##############################################################################################################
def retrieve_model_av_std(summary):
    """
        This function takes as argument the dicitonary given by functions in baselines_scripts and returns model averages and standard deviations of accuracies and weightsself.

        :param summary: dictionary of model outputs
        :reuturns: dictionary of summary of summary statistics
    """

    exps_ = [s[:-2] for s in list(summary.keys())]
    exps = set(exps_)

    NUM_REP = int(len(exps_) / len(exps))
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

##############################################################################################################
def sample_trn_test_index(index,split=2.0/3,mode='final',group=20,
    options={'submode': 'interpolation', 'samples_per_interval': 1, 'temporal_inteval': '1H'}):

    """
        Given a dataframe index, sample indexes for different training and test splits. It is possible to create different test subgroups to test temporal consistency of models.

        :param index: dataframe index from which sample training and test locations from (required to return a dataframe as well, without losing the original indexing)
        :param split: float, training to test datapoints ratios.
        :param group: int or 'all', providing the number of samples in each test subgroup, if needed, and for other things.
        :param mode: string
            |    - 'prediction' : first split used for training, rest for testing
            |    - 'interpolation': pure random sampling

            |    - 'middle' : samples equal groups for training and groups for testing, with size specified by group, independent training are all indexed by 1 and the tests are independent groups with label l in {2,...}, uniformly distributed
            |    - 'initial' : recursively uses 'final' but on inverted indexes
            |    - 'training_shifted' : indexes training points in temporally shifted lags, with group specifying how many points _before_ the training set is sampled.
            |       e.g. 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 where the first 6 "2"s are def in group
        :param options: additional options, for now only for temporal subsampling.

        :returns: a dataframe with an index column, with integres i denoting wether a datapoint is training (i=1) or test (i=2,...)
    """

    # data size
    s0 = len(index)

#    mode = mode[0]
    # initial indexing  
    ind_set = np.ones((s0,1))

    # how many training (s1) and test (s2)
    s1 = int(np.floor(s0*split))
    s2 = s0-s1

    if mode == 'prediction':
        if group == 'all':
            ind_set[(s1+1):] = 2
        else:
            for bl, ii in enumerate(range(0,s2,group)):
                ind_set[(s1+1+ii):(s1+ii+1+group)] = bl+2

    elif mode=='interpolation':
        ind_set[s1+1:] = 2
        np.random.shuffle(ind_set)

    elif mode=='temporal_subsampling':

        sub_mode = options['submode'] # interpolation or prediction
        per_temp = options['samples_per_interval'] # how many samples to take per unit of temporal int
        time_int = options['temporal_interval']

        ind_set = np.zeros((s0,1)) # override initial indexing...
        ind_set = pd.DataFrame(ind_set, index=index)

        init = ind_set.index[0].round(time_int[-1])
        endit = ind_set.index[-1].round(time_int[-1])

        samp_ints = pd.date_range(start=init, end=endit, freq=time_int)

        tr_ts_splits = np.ones((len(samp_ints),1))
        tr_ts_splits[int(np.floor(len(samp_ints)*split))+1:] = 2

        if sub_mode == 'interpolation':
            np.random.shuffle(tr_ts_splits)
        elif sub_mode == 'prediction':
            _ = None #do nothing

        # sample per_temp examples every time_int. time_ints are split randomly for training and testing
        b = []

        for t_i, t_ in enumerate(samp_ints[0:]):
            to_time = t_+pd.to_timedelta(time_int)-pd.to_timedelta('1S')
            # print(t_,to_time)
            sub_s = ind_set.loc[t_:to_time,:]

            if per_temp > 1:
                samp_frac = per_temp

            elif (per_temp>0)&(per_temp<=1):
                samp_frac = int(np.floor(per_temp*sub_s.shape[0]))

            # print(per_temp, samp_frac)

            if sub_s.shape[0] > samp_frac:
                inds_to_sam = sub_s.sample(n=samp_frac).sort_index().index
                # print(inds_to_sam)
                ind_set.loc[inds_to_sam] = tr_ts_splits[t_i][0]
                # print(ind_set.loc[inds_to_sam])

        ind_set = ind_set.values

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
