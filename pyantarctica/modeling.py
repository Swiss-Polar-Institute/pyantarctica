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
from sklearn.metrics import euclidean_distances

##############################################################################################################
# def retrieve_model_av_std(summary):
#     """
#         This function takes as argument the dicitonary given by functions in baselines_scripts and returns model averages and standard deviations of accuracies and weights.
#
#         :param summary: dictionary of model outputs
#         :reuturns: dictionary of summary of summary statistics
#     """
#
#     exps_ = [s[:-2] for s in list(summary.keys())]
#     exps = set(exps_)
#
#     NUM_REP = int(len(exps_) / len(exps))
#     results = {}
#     for name_ in exps:
#     #     print(name_)
#         results[name_] = {}
#
#         init_ = True
#         for nre in range(0,NUM_REP,1):
#             sub_res = summary[name_ + '_' + str(nre)]
#             if init_:
#                 for sub_, val_ in sub_res.items():
#                         exec(sub_ + '= []' )
#                         init_ = False
#
#             for sub_, val_ in sub_res.items():
#     #             print('-> ', sub_,val_)
#                 exec(sub_+'.append(val_)')
#
#         # for sub_ in sub_res:
#         #     if ('_gt' not in sub_) and ('_hat' not in sub_) :
#         #         exec(sub_ + '= np.array(' + sub_ + ')')
#
#         for sub_ in sub_res:
#             if ('_gt' not in sub_) and ('_hat' not in sub_) and ('_std' not in sub_):
#                 exec(sub_ + '= np.array(' + sub_ + ')')
#                 exec('results[name_][sub_] = np.append(np.mean([' + sub_ + '], axis=1), np.std([' + sub_ + '], axis=1),axis=0)')
#             else:
#                 exec('results[name_][sub_] =' + sub_)
#
#     return results

##############################################################################################################
def sample_trn_test_index(index,split=2.0/3,mode='final', group=20, options={'submode': 'interpolation', 'samples_per_interval': 1, 'temporal_inteval': '1H'}):

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
            _ = None #do nothing for now, this is a placeholder

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

##############################################################################################################
def compute_mutual_information(X,Y,nbins=128,sigma_smooth=2):
    from scipy import ndimage

    # 1: pair data and drop nans
    V = pd.concat([X,Y], axis=1).dropna()
    v1, v2 = (V.iloc[:,0], V.iloc[:,1]) # could add +1 to both to avoid 0 counts.

    pxy, xedges, yedges = np.histogram2d(v1, v2, bins=nbins, normed=True, weights=None)

    # smooth the 2d hist, to be more distribution-y
    ndimage.gaussian_filter(pxy, sigma=sigma_smooth, mode='constant', output=pxy)

    if 0:
        plt.figure()
        plt.imshow(pxy, origin='low', aspect='auto',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    # compute marginal histograms (achtung zeros when dividing)
    pxy = pxy + 1E-12
    norm = np.sum(pxy)
    pxy = pxy / norm
    py = np.sum(pxy, axis=0).reshape((-1, nbins))
    px = np.sum(pxy, axis=1).reshape((nbins, -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.

    return - np.sum(py * np.log(py)) - np.sum(px * np.log(px)) + np.sum(pxy * np.log(pxy))

##############################################################################################################
def compute_normalized_mutual_information():
    return None
    #NMI[i1,i2] = ((np.sum(py * np.log(py)) + np.sum(px * np.log(px))) / np.sum(pxy * np.log(pxy))) - 1

##############################################################################################################
def compute_approximate_HSIC(X,Y, ncom=100, gamma=[None, None], ntrials=100, random_state=1, sigma_prior = 1):
    """
        Using approximations, computes the HSIC score between two different data series. Apprixmations are subsampling for the RBF kernel bandwidth selection and random kitchen sinks to approximate kernels (adn therefore directly using inner products to estimate the cross-covariance operator in the approximate RKHS)

        :param X, Y: (mutlivariate) data series
        :param ncom: number of components to use in the random kernel approximation
        :param gamma: bandwidth for the RBF kernels
        :param ntrials: number of trials, on which HSIC is averaged
        :param random_state: set initial random state for reproducibility
        :param sigma_prior: scaling for the sigmas
    """

    from sklearn.kernel_approximation import RBFSampler
    import random
    if random_state is not None:
        random.seed(random_state)

    def centering(K):
        """
            center kernel matrix
        """
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        Q = I - unit/n
        return np.dot(np.dot(Q, K), Q)

    def rbf(X, sigma=None):
        """
            define RBF kernel + its parameter
        """
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = np.sqrt(mdist)
        KX *= - 0.5 / sigma / sigma
        np.exp(KX, KX)
        return KX

    if gamma[0] is None:

        if X.shape[0] > 1000:
            yy = np.random.choice(len(X),1000)
            x_ = X[yy]
            del yy
        else:
            x_ = X

        GX = np.dot(x_, x_.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        mdist = np.median(KX[KX != 0])
        gamma[0] = 1/(np.sqrt(sigma_prior*mdist)**2)
        del GX, KX, mdist

    if gamma[1] is None:
        if Y.shape[0] > 1000:
            yy = np.random.choice(len(Y),1000)
            y_ = Y[yy]
            del yy
        else:
            y_ = Y

        GY = np.dot(y_, y_.T)
        KY = np.diag(GY) - GY + (np.diag(GY) - GY).T
        mdist = np.median(KY[KY != 0])
        gamma[1] = 1/(np.sqrt(sigma_prior*mdist)**2)
        del GY, KY, mdist

    hs_a = 0
    rbf_feature_x = RBFSampler(gamma=gamma[0], random_state=random_state, n_components=ncom)
    rbf_feature_y = RBFSampler(gamma=gamma[1], random_state=random_state, n_components=ncom)

    for trial in range(ntrials):
        if (X.shape[0] < 1)|(Y.shape[0] < 1):
            continue

        X_f = rbf_feature_x.fit_transform(X)
        X_f -= np.mean(X_f,axis=0)
        Y_f = rbf_feature_y.fit_transform(Y)
        Y_f -= np.mean(Y_f,axis=0)

        A = X_f.T.dot(Y_f)
        B = Y_f.T.dot(X_f)
        C = A.dot(B)
        hs_a += 1/X_f.shape[0]**2 * np.trace(C)

    return hs_a / ntrials

##############################################################################################################
def dependency_measures_per_dataset(series_1, series_2):
    from tqdm import tqdm
    file_string = 'part_waves'

    COR = np.zeros((series_1.shape[1],series_2.shape[1]))
    MI = np.zeros((series_1.shape[1],series_2.shape[1]))
    HSIC = np.zeros((series_1.shape[1],series_2.shape[1]))
    NSAMP = np.zeros((series_1.shape[1],series_2.shape[1]))

    for i1, pa1 in enumerate(series_1.columns.tolist()):
        print(i1, end=' ')
        for i2, pa2 in enumerate(series_2.columns.tolist()):
    #         print('.',end='')
            if set(series_1.columns.tolist())==set(series_2.columns.tolist()): # True:
                if i2 >= i1:
                    # print(pa1, pa2, end=' ')
                    # 1: pair data and drop nans
                    V = pd.concat([series_1.iloc[:,i1],series_2.iloc[:,i2]], axis=1).dropna()
                    v1, v2 = (V.iloc[:,0], V.iloc[:,1]) # could add +1 to both to avoid 0 counts.

                    rho = np.corrcoef(V.transpose())#.iloc[:,0], V.iloc[:,1])
                    COR[i1,i2] = rho[0,1]
                    MI[i1,i2] = compute_mutual_information(v1,v2,nbins=128,sigma_smooth=2)
                    HSIC[i1,i2] = compute_approximate_HSIC(v1.values.reshape(-1,1),v2.values.reshape(-1,1), ncom=100, gamma=[None, None], ntrials=100, random_state=1, sigma_prior=1)
            else:
                V = pd.concat([series_1.iloc[:,i1],series_2.iloc[:,i2]], axis=1).dropna()
                v1, v2 = (V.iloc[:,0], V.iloc[:,1]) # could add +1 to both to avoid 0 counts.

                rho = np.corrcoef(V.transpose())#.iloc[:,0], V.iloc[:,1])
                COR[i1,i2] = rho[0,1]
                MI[i1,i2] = compute_mutual_information(v1,v2,nbins=128,sigma_smooth=2)
                HSIC[i1,i2] = compute_approximate_HSIC(v1.values.reshape(-1,1),v2.values.reshape(-1,1), ncom=100, gamma=[None, None], ntrials=5, random_state=1, sigma_prior = 1)
                NSAMP[i1,i2] = V.shape[0]

    if set(series_1.columns.tolist())==set(series_2.columns.tolist()): # True:
        COR = (COR + COR.T) - np.diag(np.diag(COR))
        MI = (MI + MI.T) - np.diag(np.diag(MI))
        HSIC = (HSIC + HSIC.T) - np.diag(np.diag(HSIC))
        NSAMP = (NSAMP + NSAMP.T) - np.diag(np.diag(NSAMP))

    return COR, MI, HSIC, NSAMP

##############################################################################################################
def CV_smooth_weight_regression(data, labels, inds, opts):
    """
        Cross-validate smooth ridge regression
        CV_MEASURE: val_R2, trn_R2, minLoss
    """

    if not opts['CV_MEASURE']:
        opts['CV_MEASURE'] = 'val_R2'

    # if opts['MODEL'] == 'smooth-linear-ridge-regression':
    #     opts['KPAR_CV'] = []
    # elif opts['MODEL'] == 'smooth-linear-bayesian-regression':
    #     opts['KPAR_CV'] = []
    print('KPAR_CV', opts['KPAR_CV'])
    if opts['KPAR_CV'] == 'distance_mode':
        D = euclidean_distances(data.dropna(), squared=False)
        # print('D_feats =', data.dropna().columns.tolist())
        D = D.reshape(-1,1)
        D = D[D!=0]
        opts['KPAR_CV'] = [np.median(D)]

        # counts,bins = np.histogram(D,bins=100)
        # opts['KPAR_CV'] = [bins[np.argmax(counts)]]

    trerr = np.zeros((len(opts['PAR_1_CV'])*len(opts['PAR_2_CV'])*len(opts['KPAR_CV']), opts['TASKS']))
    vaerr = np.zeros((len(opts['PAR_1_CV'])*len(opts['PAR_2_CV'])*len(opts['KPAR_CV']), opts['TASKS']))

    minlo = np.zeros((len(opts['PAR_1_CV'])*len(opts['PAR_2_CV'])*len(opts['KPAR_CV']), 1))
    pars = np.zeros((len(opts['PAR_1_CV'])*len(opts['PAR_2_CV'])*len(opts['KPAR_CV']), 3))
    print()
    iit  = pars.shape[0]
    count = 0
    for p1 in opts['PAR_1_CV']:
        for p2 in opts['PAR_2_CV']:
            for kpar in opts['KPAR_CV']:

                opts['par1'] = p1
                opts['par2'] = p2
                opts['kpar'] = kpar

                if opts['MODEL'] == 'smooth_weight_ridge_regression':
                    W_mtl, loss, inds, stats, y_hat = smooth_weight_ridge_regression(data, labels, inds, opts)#, ITERS=100, THRESH=1e-6):
                elif opts['MODEL'] == 'smooth_weight_kernel_ridge_regression':
                    W_mtl, loss, inds, stats, y_hat = smooth_weight_kernel_ridge_regression(data, labels, inds, opts)
                elif opts['MODEL'] == 'approximate_smooth_weight_kernel_ridge_regression':
                    W_mtl, loss, inds, stats, y_hat = approximate_smooth_weight_kernel_ridge_regression(data, labels, inds, opts)
                elif opts['MODEL'] == 'bayesian_smooth_weight_ridge_regression':
                    W_mtl, loss, inds, stats, y_hat = bayesian_smooth_weight_ridge_regression(data, labels, inds, opts)
                elif opts['MODEL'] == 'ssmooth_weight_approximate_gaussian_process_regression':
                    W_mtl, loss, inds, stats, y_hat = smooth_weight_approximate_gaussian_process_regression(data, labels, inds, opts)

                trerr[count, :] = stats['tr_R2']
                vaerr[count, :] = stats['va_R2']
                minlo[count, 0] = loss[-1]
                pars[count, :] = [p1,p2,kpar]

                print(f"{count+1} / {iit}")
                # p1 {p1}, p2 {p2}, kpar {kpar}, loss {loss[-1]}, trerr {0.01 * np.sum(stats['tr_R2'])}, valerr {0.01 * np.sum(stats['va_R2'])}")
                count += 1


    if opts['CV_MEASURE'] == 'trn_R2':
        p1,p2,kpar = pars[np.argmax(np.sum(vaerr,axis=1)),:]

    elif opts['CV_MEASURE'] == 'val_R2':
        p1,p2,kpar = pars[np.argmax(np.sum(trerr,axis=1)),:]

    elif opts['CV_MEASURE'] == 'minLoss':
        p1,p2,kpar = pars[np.argmin(minlo),:]

    #print(f'p1 {p1}, p2 {p2}, kpar {kpar}')
    return p1, p2, kpar

##############################################################################################################
def smooth_weight_ridge_regression(data, labels, inds, opts):#, ITERS=100, THRESH=1e-6):
    """
        Define a multitask regression problem in which tasks are locally smooth (e.g. bin size prediction), and introduce a penalty in which weights of regressors of related tasks are encouraged to be similar.
    """

    from sklearn.metrics import mean_squared_error, r2_score
    # from sklearn.model_selection import KFold

    def retrieve_neigh_norm(W,ind_w,ss):
        L = W.shape[1]
        hs = int(np.floor(ss/2))
        if ind_w < hs:
            W_subs = W[:,0:(ind_w + hs + 1)]
            # W_subs = W[:,np.setdiff1d(range(0,(ind_w+hs+1),1),ind_w)]

        elif ind_w > (L-hs-1):
            W_subs = W[:,(ind_w-hs):L]
            # W_subs = W[:,np.setdiff1d(range((ind_w-hs),L,1),ind_w)]

        else:
            W_subs = W[:,(ind_w-hs):(ind_w+hs+1)]
            # W_subs = W[:,np.setdiff1d(range((ind_w-hs),(ind_w+hs+1),1),ind_w)]

        return 1/(len(W_subs)) * np.sum(W_subs, axis=1)

    D = data.shape[1]
    T = labels.shape[1]

    # print(opts)

    par1 = opts['par1']
    par2 = opts['par2']
    # tr_ts_split = opts['tr_ts_split']

    W_old = np.ones((D,T))
    W = np.random.rand(D,T)

    # Not so easy to use defaults... see if something can be done to use CV
    # k_fold = KFold(n_splits=opts['KFOLD'])
    # for train_indices, test_indices in k_fold.split(X):

    # init stuff
    k = 0
    epsi = 1000
    loss = []
    while (epsi > opts['THRESH'])&(k < opts['ITERS']):
        # schedule = np.random.permutation(range(T))
        schedule = range(T)
        W_old = W.copy()
        for ind_w in schedule:
            X_Y = data.assign(y=labels.iloc[:,ind_w]).copy()
            train_X = X_Y.iloc[inds[:,ind_w] == 1, :-1]
            train_Y = X_Y.iloc[inds[:,ind_w] == 1, -1]
            N = train_X.shape[0]

            W_mt_n = retrieve_neigh_norm(W,ind_w, opts['WIN_SIZE'])

            # W_mt_n = 1/T * np.sum(W[:,~ind_w],axis=1)
            # A = 1/N * np.dot(train_X.T,train_X) + 1/N * par1*np.eye(D) + 1/T * par2*np.eye(D)
            A = np.dot(train_X.T,train_X) + par1*np.eye(D) + par2*np.eye(D)
            A = np.linalg.inv(A)

            # B = 1/N * np.dot(train_X.T,train_Y) + 1/T * par2*W_mt_n
            B = np.dot(train_X.T,train_Y) + par2*W_mt_n

            W[:,ind_w] = np.dot(A,B)

        epsi = np.abs(np.sum(np.sum(W-W_old)))
        # print(f'iter {k}, size W {W.shape}, loss {epsi}')

        loss.append(epsi)
        k += 1
    # print(f'iter {k}, size W {W.shape}, loss {epsi}')

    y_hat = np.zeros((opts['DATA_SIZE'][0],T))
    stats = {}
    stats['tr_RMSE'] = []#np.inf*np.ones((T))
    stats['tr_R2'] = []#np.inf*np.ones((T))
    stats['va_RMSE'] = []#np.inf*np.ones((T))
    stats['va_R2'] = []#np.inf*np.ones((T))

    # if opts['VAL_MODE']:
    for ind_w in range(T):
        X_Y = data.assign(y=labels.iloc[:,ind_w]).copy()
        trn_X = X_Y.iloc[inds[:,ind_w] == 1, :-1]

        trn_Y = X_Y.iloc[inds[:,ind_w] == 1, -1]
        pred_tr_Y = np.dot(trn_X,W[:,ind_w])
        if np.sum(np.isinf(pred_tr_Y))>0:
            print(f'sum isinf: {np.sum(np.isinf(pred_tr_Y))}')
            print(f'sum isna: {np.sum(np.isnan(pred_tr_Y))}')
            stats['tr_RMSE'].append(np.inf)
            stats['tr_R2'].append(-np.inf)
        else:
            stats['tr_RMSE'].append(np.sqrt(mean_squared_error(trn_Y,pred_tr_Y)))
            stats['tr_R2'].append(r2_score(trn_Y,pred_tr_Y))

        y_hat[inds[:,ind_w] == 1, ind_w] = pred_tr_Y

        if np.any(inds[:,ind_w] == 2):

            val_X = X_Y.iloc[inds[:,ind_w] == 2, :-1]
            val_Y = X_Y.iloc[inds[:,ind_w] == 2, -1]
            pred_va_Y = np.dot(val_X,W[:,ind_w])
            if np.sum(np.isinf(pred_tr_Y))>0:
                print(f'VAL: sum isinf: {np.sum(np.isinf(pred_tr_Y))}')
                print(f'VAL: sum isna: {np.sum(np.isnan(pred_tr_Y))}')
                stats['va_RMSE'].append(np.inf)
                stats['va_R2'].append(-np.inf)
            else:
                stats['va_RMSE'].append(np.sqrt(mean_squared_error(val_Y,pred_va_Y)))
                stats['va_R2'].append(r2_score(val_Y,pred_va_Y))


            y_hat[inds[:,ind_w] == 2, ind_w] = pred_va_Y

    return W, loss, inds, stats, y_hat

##############################################################################################################
def bayesian_smooth_weight_ridge_regression(data, labels, inds, opts):#, ITERS=100, THRESH=1e-6):
    """
        Define a Bayesian multitask regression problem in which tasks are locally smooth (e.g. bin size prediction), and introduce a penalty in which weights of regressors of related tasks are encouraged to be similar.
        See https://icml.cc/Conferences/2005/proceedings/papers/128_GaussianProcesses_YuEtAl.pdf
    """
    import scipy.stats as stats
    from sklearn.metrics import mean_squared_error, r2_score
    # from sklearn.model_selection import KFold

    D = data.shape[1]
    T = labels.shape[1]

    # par1 = opts['par1']
    # par2 = opts['par2']
    # tr_ts_split = opts['tr_ts_split']

    W_old = np.ones((D,T))
    W = np.random.rand(D,T)

    # Not so easy to use defaults... see if something can be done to use CV
    # k_fold = KFold(n_splits=opts['KFOLD'])
    # for train_indices, test_indices in k_fold.split(X):
    tau = opts['par1']
    pi = opts['par2']

    # C_ = data.dropna().sample(n=100)
    # C_ = 1/(100-1)*C_.T.dot(C_)

    # sample priors from hyperprior (NIW)
    C_w = stats.invwishart.rvs(tau, np.eye(D))
    # print(np.linalg.eig(C_w))
    mu_w = np.random.multivariate_normal(np.zeros((D)), 1/pi * C_w) # np.ones((D,1))
    # print(mu_w)
    sigma = 1

    k = 0
    epsi = 1000
    loss = []

    while (epsi > opts['THRESH'])&(k < opts['ITERS']):
        # schedule = np.random.permutation(range(T))

        # E-step:
        schedule = range(T)
        W_old = W.copy()
        C_w_temp = 0
        w_l_ce_temp = 0
        sigma_l = 0
        n_l = 0
        for ind_w in schedule:
            # print(ind_w)
            X_Y = data.assign(y=labels.iloc[:,ind_w]).copy()
            train_X = X_Y.iloc[inds[:,ind_w] == 1, :-1]
            train_Y = X_Y.iloc[inds[:,ind_w] == 1, -1]
            N = train_X.shape[0]
            A = 1/sigma * np.dot(train_X.T,train_X) + np.linalg.inv(C_w)
            A = np.linalg.inv(A)
            # print(A)
            C_w_temp += A

            # print(np.dot(train_X.T,train_Y).T, np.matmul(np.linalg.inv(C_w), mu_w))
            B = 1/sigma * np.dot(train_X.T,train_Y).T + np.matmul(mu_w.T, np.linalg.inv(C_w))
            W[:,ind_w] = np.dot(A,np.squeeze(B))

            w_ = W[:,ind_w] - mu_w
            w_l_ce_temp += np.dot(w_,w_.T)

            n_l += N
            sigma_l += np.sum((train_Y - train_X.dot(W[:,ind_w]))**2) + np.trace(train_X.dot(C_w.dot(train_X.T)))
            # print(mu_w)
            #

        # M-step
        mu_w = 1/(pi + T) * np.sum(W,axis=1)
        C_w =  1/(tau + T) * (pi*np.dot(mu_w,mu_w.T) + tau*np.eye(D) + w_l_ce_temp) # + tau*np.eye(D)
        sigma = 1/n_l * sigma_l


        epsi = np.abs(np.sum(np.sum(W-W_old)))
        loss.append(epsi)
        # print(f'iter {k}, size W {W.shape}, loss {epsi}')
        k += 1
    # print(f'iter {k}, cost {epsi})
    print(f'iter {k}, size W {W.shape}, loss {epsi}')

    y_hat = np.zeros((opts['DATA_SIZE'][0],T))
    stats = {}
    stats['tr_RMSE'] = []#np.inf*np.ones((T))
    stats['tr_R2'] = []#np.inf*np.ones((T))
    stats['va_RMSE'] = []#np.inf*np.ones((T))
    stats['va_R2'] = []#np.inf*np.ones((T))

    # if opts['VAL_MODE']:
    for ind_w in range(T):
        X_Y = data.assign(y=labels.iloc[:,ind_w]).copy()
        trn_X = X_Y.iloc[inds[:,ind_w] == 1, :-1]
        trn_Y = X_Y.iloc[inds[:,ind_w] == 1, -1]
        pred_tr_Y = np.dot(trn_X,W[:,ind_w])

        stats['tr_RMSE'].append(np.sqrt(mean_squared_error(trn_Y,pred_tr_Y)))
        stats['tr_R2'].append(r2_score(trn_Y,pred_tr_Y))

        y_hat[inds[:,ind_w] == 1, ind_w] = pred_tr_Y

        if np.any(inds[:,ind_w] == 2):

            val_X = X_Y.iloc[inds[:,ind_w] == 2, :-1]
            val_Y = X_Y.iloc[inds[:,ind_w] == 2, -1]
            pred_va_Y = np.dot(val_X,W[:,ind_w])

            stats['va_RMSE'].append(np.sqrt(mean_squared_error(val_Y,pred_va_Y)))
            stats['va_R2'].append(r2_score(val_Y,pred_va_Y))

            y_hat[inds[:,ind_w] == 2, ind_w] = pred_va_Y

    return W, loss, inds, stats, y_hat

##############################################################################################################
def smooth_weight_kernel_ridge_regression(data, labels, inds, opts):#, ITERS=100, THRESH=1e-6):
    """
        Define a multitask regression problem in which tasks are locally smooth (e.g. bin size prediction), and introduce a penalty in which weights of regressors of related tasks are encouraged to be similar.
    """

    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.metrics.pairwise import rbf_kernel

    def retrieve_neigh_norm(W,ind_w,ss):
        L = W.shape[1]
        hs = int(np.floor(ss/2))
        if ind_w < hs:
            W_subs = W[:,0:(ind_w + hs + 1)]
            # W_subs = W[:,np.setdiff1d(range(0,(ind_w+hs+1),1),ind_w)]

        elif ind_w > (L-hs-1):
            W_subs = W[:,(ind_w-hs):L]
            # W_subs = W[:,np.setdiff1d(range((ind_w-hs),L,1),ind_w)]

        else:
            W_subs = W[:,(ind_w-hs):(ind_w+hs+1)]
            # W_subs = W[:,np.setdiff1d(range((ind_w-hs),(ind_w+hs+1),1),ind_w)]

        return 1/(ss**2) * np.sum(W_subs, axis=1)

    D = data.shape[1]
    T = labels.shape[1]

    par1 = opts['par1']
    par2 = opts['par2']
    kpar = opts['kpar']
    # tr_ts_split = opts['tr_ts_split']

    W_old = np.ones((D,T))
    W = np.random.rand(D,T)


    # init stuff
    k = 0
    epsi = np.Inf
    loss = []
    while (epsi > opts['THRESH']) & (k < opts['ITERS']):
        # schedule = np.random.permutation(range(T))
        schedule = range(opts['TASKS'])
    W_old = W.copy()
    for ind_w in schedule:

        X_Y = data.assign(y=labels.iloc[:,ind_w]).copy()
        trn_X = X_Y.iloc[ind_mat[:,ind_w] == 1, :-1]
        trn_Y = X_Y.iloc[ind_mat[:,ind_w] == 1, -1]

        K = rbf_kernel(trn_X, gamma=1./opts['kpar']**2)
        W_mt_n = retrieve_neigh_norm(W[ind_mat[:,ind_w] == 1,:], ind_w, opts['WIN_SIZE'])

        A = K + opts['par1']*np.eye(np.sum(ind_mat[:,ind_w] == 1)) + opts['par2']*np.eye(np.sum(ind_mat[:,ind_w] == 1))
        A = np.linalg.inv(A)

        B = opts['par2']*W_mt_n + trn_Y
        W[ind_mat[:,ind_w] == 1,ind_w] = np.dot(A,B)

        epsi = np.abs(np.sum(np.sum(W-W_old)))
        # print(f'iter {k}, size W {W.shape}, loss {epsi}')

        loss.append(epsi)
        k += 1

    print(f'iter {k}, size W {W.shape}, loss {epsi}')

    y_hat = np.zeros((opts['DATA_SIZE'][0],T))
    stats = {}
    stats['tr_RMSE'] = []#np.inf*np.ones((T))
    stats['tr_R2'] = []#np.inf*np.ones((T))
    stats['va_RMSE'] = []#np.inf*np.ones((T))
    stats['va_R2'] = []#np.inf*np.ones((T))

    # if opts['VAL_MODE']:
    for ind_w in range(opts['TASKS']):
        X_Y = data.assign(y=labels.iloc[:,ind_w]).copy()
        trn_X = X_Y.iloc[ind_mat[:,ind_w] == 1, :-1]
        trn_Y = X_Y.iloc[ind_mat[:,ind_w] == 1, -1]

        K = rbf_kernel(trn_X, gamma=1./sigma**2)
        pred_tr_Y = np.dot(K,W[ind_mat[:,ind_w] == 1,ind_w])
        if np.sum(np.isinf(pred_tr_Y))>0:
            print(f'sum isinf: {np.sum(np.isinf(pred_tr_Y))}')
            print(f'sum isna: {np.sum(np.isnan(pred_tr_Y))}')
            stats['tr_RMSE'].append(np.inf)
            stats['tr_R2'].append(-np.inf)
        else:
            stats['tr_RMSE'].append(np.sqrt(mean_squared_error(trn_Y,pred_tr_Y)))
            stats['tr_R2'].append(r2_score(trn_Y,pred_tr_Y))


        y_hat[ind_mat[:,ind_w] == 1, ind_w] = pred_tr_Y

        if np.any(ind_mat[:,ind_w] == 2):
            val_X = X_Y.iloc[ind_mat[:,ind_w] == 2, :-1]
            val_Y = X_Y.iloc[ind_mat[:,ind_w] == 2, -1]

            Kt = rbf_kernel(val_X,trn_X, gamma=1./sigma**2)
            pred_va_Y = np.dot(Kt,W[ind_mat[:,ind_w] == 1,ind_w])

            if np.sum(np.isinf(pred_tr_Y))>0:
                print(f'sum isinf: {np.sum(np.isinf(pred_tr_Y))}')
                print(f'sum isna: {np.sum(np.isnan(pred_tr_Y))}')
                stats['va_RMSE'].append(np.inf)
                stats['va_RMSE'].append(-np.inf)
            else:
                stats['va_RMSE'].append(np.sqrt(mean_squared_error(val_Y,pred_va_Y)))
                stats['va_R2'].append(r2_score(val_Y,pred_va_Y))

            y_hat[ind_mat[:,ind_w] == 2, ind_w] = pred_va_Y

    return W, loss, inds, stats, y_hat

##############################################################################################################
def approximate_smooth_weight_kernel_ridge_regression(data, labels, ind_mat, opts):#, ITERS=100, THRESH=1e-6):
    """
        Define a multitask regression problem in which tasks are locally smooth (e.g. bin size prediction), and introduce a penalty in which weights of regressors of related tasks are encouraged to be similar. Instead of optimizing the dual and using explicit kernel matrices, random kitchen sinks style kernel approximations are used.
    """

    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.kernel_approximation import RBFSampler
    D = opts['APPROXIMATION_DIM']
    sm = RBFSampler(gamma=1./opts['kpar']**2,n_components=D,random_state=666)

    def retrieve_neigh_norm(W,ind_w,ss):
        L = W.shape[1]
        hs = int(np.floor(ss/2))
        if ind_w < hs:
            W_subs = W[:,0:(ind_w + hs + 1)]
        elif ind_w > (L-hs-1):
            W_subs = W[:,(ind_w-hs):L]
        else:
            W_subs = W[:,(ind_w-hs):(ind_w+hs+1)]
        return 1/(ss) * np.sum(W_subs, axis=1)

    W = np.random.rand(D,opts['TASKS'])

    # init stuff
    k = 0
    epsi = np.Inf
    loss = []
    while (epsi > opts['THRESH']) & (k < opts['ITERS']):
        # schedule = np.random.permutation(range(T))
        schedule = range(opts['TASKS'])
        W_old = W.copy()
        t_loss = 0
        t_norm_w = 0

        for ind_w in schedule:
            X_Y = data.assign(y=labels.iloc[:,ind_w]).copy()
            trn_X = sm.fit_transform(X_Y.iloc[ind_mat[:,ind_w] == 1, :-1])
            trn_Y = X_Y.iloc[ind_mat[:,ind_w] == 1, -1]
            N = np.sum(ind_mat[:,ind_w] == 1)

            W_mt_n = retrieve_neigh_norm(W, ind_w, opts['WIN_SIZE'])

            A = np.dot(trn_X.T,trn_X) + opts['par1']*np.eye(D) + opts['par2']*np.eye(D)
            A = np.linalg.inv(A)
            B = np.dot(trn_X.T,trn_Y) + opts['par2']*W_mt_n
            W[:,ind_w] = np.dot(A,B)

            t_loss += 1/N*np.sum((trn_Y.values - np.dot(trn_X,W[:,ind_w]))**2) # + np.linalg.norm(W[:,ind_w])
            t_norm_w += np.linalg.norm(W[:,ind_w])

        emp_err = 1/opts['TASKS'] * t_loss
        epsi = np.abs(np.sum(np.sum(W-W_old)))
        loss.append(emp_err)
        # print(f'iter {k}, size W {W.shape}, conv {epsi}, emp error {emp_err}, mean norm w {1/opts["TASKS"] * t_norm_w}')

        k += 1

    if k == opts['ITERS']:
        print(f'converged in iter {k}, size W {W.shape}, conv {epsi}, emp error {emp_err}, mean norm w {1/opts["TASKS"] * t_norm_w}')

    y_hat = np.zeros((opts['DATA_SIZE'][0],opts['TASKS']))
    pred_stats = {}
    pred_stats['tr_RMSE'] = []#np.inf*np.ones((T))
    pred_stats['tr_R2'] = []#np.inf*np.ones((T))
    pred_stats['va_RMSE'] = []#np.inf*np.ones((T))
    pred_stats['va_R2'] = []#np.inf*np.ones((T))

    # if opts['VAL_MODE']:
    for ind_w in range(opts['TASKS']):
        X_Y = data.assign(y=labels.iloc[:,ind_w]).copy()
        trn_X = sm.fit_transform(X_Y.iloc[ind_mat[:,ind_w] == 1, :-1])
        trn_Y = X_Y.iloc[ind_mat[:,ind_w] == 1, -1]

        pred_tr_Y = np.dot(trn_X,W[:,ind_w])

        pred_stats['tr_RMSE'].append(np.sqrt(mean_squared_error(trn_Y,pred_tr_Y)))
        pred_stats['tr_R2'].append(r2_score(trn_Y,pred_tr_Y))

        y_hat[ind_mat[:,ind_w] == 1, ind_w] = pred_tr_Y

        if np.any(ind_mat[:,ind_w] == 2):
            val_X = sm.fit_transform(X_Y.iloc[ind_mat[:,ind_w] == 2, :-1])
            val_Y = X_Y.iloc[ind_mat[:,ind_w] == 2, -1]

            pred_va_Y = np.dot(val_X,W[:,ind_w])

            pred_stats['va_RMSE'].append(np.sqrt(mean_squared_error(val_Y,pred_va_Y)))
            pred_stats['va_R2'].append(r2_score(val_Y,pred_va_Y))

            y_hat[ind_mat[:,ind_w] == 2, ind_w] = pred_va_Y

    return W, loss, ind_mat, pred_stats, y_hat

##############################################################################################################
def smooth_weight_approximate_gaussian_process_regression(data, labels, ind_mat, opts):#, ITERS=100, THRESH=1e-6):
    """
        Define a Bayesian multitask regression problem in which tasks are locally smooth (e.g. bin size prediction), and introduce a penalty in which weights of regressors of related tasks are encouraged to be similar.
        See https://icml.cc/Conferences/2005/proceedings/papers/128_GaussianProcesses_YuEtAl.pdf
    """

    import scipy.stats as stats
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.kernel_approximation import RBFSampler

    D = opts['approximation_dim']

    sm = RBFSampler(gamma=1./opts['kpar']**2,n_components=D,random_state=666)

    T = opts['TASKS']
    W = np.random.rand(D,T)
    tau = opts['par1']
    pi = opts['par2']

    C_w = stats.invwishart.rvs(tau, np.eye(D))
    mu_w = np.random.multivariate_normal(np.zeros((D)), 1/pi * C_w) # np.ones((D,1))

    sigma = D / T
    k = 0
    epsi = 1000
    loss = []

    while (epsi > opts['THRESH'])&(k < opts['ITERS']):
        # E-step:
        schedule = range(T)
        W_old = W.copy()
        C_w_temp = 0
        w_l_ce_temp = 0
        sigma_l = 0
        n_l = 0
        t_loss = 0
        t_norm_w = 0
        for ind_w in schedule:
            # print(ind_w)
            X_Y = data.assign(y=labels.iloc[:,ind_w]).copy()
            trn_X = sm.fit_transform(X_Y.iloc[ind_mat[:,ind_w] == 1, :-1])
            trn_Y = X_Y.iloc[ind_mat[:,ind_w] == 1, -1]

            N = trn_X.shape[0]
            A = 1/sigma * np.dot(trn_X.T,trn_X) + np.linalg.inv(C_w)
            A = np.linalg.inv(A)
            # print(A)
            C_w_temp += A

            B = 1/sigma * np.dot(trn_X.T,trn_Y).T + np.dot(mu_w.T, np.linalg.inv(C_w))
            W[:,ind_w] = np.dot(A,np.squeeze(B))

            w_ = W[:,ind_w] - mu_w
            w_l_ce_temp += np.dot(w_,w_.T)

            n_l += N
            sigma_l += np.sum((trn_Y - trn_X.dot(W[:,ind_w]))**2) + np.trace(np.dot(trn_X,np.dot(C_w,trn_X.T)))

            t_loss += 1/N*np.sum((trn_Y.values - np.dot(trn_X,W[:,ind_w]))**2) # + np.linalg.norm(W[:,ind_w])
            t_norm_w += np.linalg.norm(W[:,ind_w])

        # M-step
        mu_w = 1/(pi + T) * np.sum(W,axis=1)
        C_w =  1/(tau + T) * (pi*np.dot(mu_w,mu_w.T) + tau*np.eye(D) + w_l_ce_temp) # + tau*np.eye(D)
        sigma = 1/n_l * sigma_l

        emp_err = 1/T * t_loss
        epsi = np.abs(np.sum(np.sum(W-W_old)))
        loss.append(emp_err)
        # print(f'iter {k}, size W {W.shape}, conv {epsi}, emp error {emp_err}, mean norm w {1/T * t_norm_w}')
        k += 1

        # print(f'iter {k}, size W {W.shape}, conv {epsi}, emp error {emp_err}, mean norm w {1/T * t_norm_w}')

    print(f'iter {k}, size W {W.shape}, conv {epsi}, emp error {emp_err}, mean norm w {1/T * t_norm_w}')

    y_hat = np.zeros((opts['DATA_SIZE'][0],T))
    pred_stats = {}
    pred_stats['tr_RMSE'] = []#np.inf*np.ones((T))
    pred_stats['tr_R2'] = []#np.inf*np.ones((T))
    pred_stats['va_RMSE'] = []#np.inf*np.ones((T))
    pred_stats['va_R2'] = []#np.inf*np.ones((T))

    # if opts['VAL_MODE']:
    for ind_w in range(T):
        X_Y = data.assign(y=labels.iloc[:,ind_w]).copy()
        trn_X = sm.fit_transform(X_Y.iloc[ind_mat[:,ind_w] == 1, :-1])
        trn_Y = X_Y.iloc[ind_mat[:,ind_w] == 1, -1]
        pred_tr_Y = np.dot(trn_X,W[:,ind_w])

        pred_stats['tr_RMSE'].append(np.sqrt(mean_squared_error(trn_Y,pred_tr_Y)))
        pred_stats['tr_R2'].append(r2_score(trn_Y,pred_tr_Y))

        y_hat[ind_mat[:,ind_w] == 1, ind_w] = pred_tr_Y

        if np.any(ind_mat[:,ind_w] == 2):

            val_X = sm.fit_transform(X_Y.iloc[ind_mat[:,ind_w] == 2, :-1])
            val_Y = X_Y.iloc[ind_mat[:,ind_w] == 2, -1]
            pred_va_Y = np.dot(val_X,W[:,ind_w])

            pred_stats['va_RMSE'].append(np.sqrt(mean_squared_error(val_Y,pred_va_Y)))
            pred_stats['va_R2'].append(r2_score(val_Y,pred_va_Y))

            y_hat[ind_mat[:,ind_w] == 2, ind_w] = pred_va_Y

    return W, loss, ind_mat, pred_stats, y_hat
