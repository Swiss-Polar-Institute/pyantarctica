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

import os
import pickle
import GPy
import numpy as np
import pandas as pd
import sklearn as sk

from datetime import datetime
from math import ceil
from sklearn import preprocessing
from pathlib import Path


import pyantarctica.dataset as dataset

import sklearn.gaussian_process.kernels as kernels #DotProduct, RBF, WhiteKernel
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from . import dataset, modeling

## FUNCTIONS
##############################################################################################################
def save_obj(obj, fname):
    """
        Save obj dict as pickle binary

        :param obj: dict, object to be saved
        :param fname: string, folder address and name of the archive to be saved
        :returns: nothing
    """
    fname = fname.with_suffix('.pkl')

    with open(fname, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

##############################################################################################################
def run_baselines_wave_particle_size(data, options):
    """
        Run regression model(s) on single replica of data (no random resampling, but uses indexes). Good for testing training and testing on manually specified splits

        :param data: input dataset, training and test mixed, [x,y] labels last column
        :param kwargs: Dictionary containing options.  Fields are:

        |    SEP_METHOD : ['interpolation','prediction'] -> Mode of learning / testing
        |    SEA : ['total','wind'] -> To subset wave parameters
        |    NUM_REP : N -> N random resampling of training and test sets (ONLY 'interpolation')
        |    LEG_P : [1, 2, 3] -> On which leg to predict. Treated separately by the models
        |    VARSET : STRING -> explicit subset of variable to use from the pandas dataframe (e.g. 'waves')
        |    METHODS : [ridgereg', 'pls', 'lasso', 'rbfgpr', 'rbfgprard', 'rf'] -> Regression method
        |    NORMALIZE_Y : BOOL -> whether to normalize outputs y
        |    NORMALIZE_X : BOOL -> whether to normalize inputs x
        |    SAVEFOLDER : STRING -> folder address to store results
        |    MODELNAME : STRING -> name of the model file and scores
        |    SPLIT_SIZE : FLOAT [0,1] -> percentage of training to test datapoints
        |    TRN_TEST_INDEX : DF -> list of integers containing wheter the point belongs to training (=1) or to the test set (=2). Has to be same size of data.shape[0]

        :returns: A dictionary containing weights, accuracy scores for training and test sets
    """

    SEP_METHOD = options['SEP_METHOD']
    SEA = options['SEA']
    NUM_REP = options['NUM_REP']
    LEG_P = options['LEG_P']
    VARSET = options['VARSET']
    METHODS = options['METHODS']
    NORMALIZE_Y = options['NORMALIZE_Y']
    NORMALIZE_X = options['NORMALIZE_X']
    SAVEFOLDER = options['SAVEFOLDER']
    MODELNAME = options['MODELNAME']
    SPLIT_SIZE = options['SPLIT_SIZE']
    TRN_TEST_INDEX = options['TRN_TEST_INDEX']

    #SAVE_TEXT_DUMP = kwargs['SAVE_TEXT_DUMP']
    if not os.path.isdir(SAVEFOLDER):
        os.mkdir(SAVEFOLDER)

    if os.path.exists(SAVEFOLDER / MODELNAME):
        print("file exists, overwriting")

    summ = {}
    #if SAVE_TEXT_DUMP:
    #    results = pd.DataFrame(index=[],columns=['tst_r2','tst_rmse','trn_r2','trn_rmse','n_tr','n_ts'])

    for sep_method in SEP_METHOD:
        #print(sep_method)
        for leg in LEG_P:
            # print(leg)
            for sea in SEA:
                for varset in VARSET:
                    for meth in METHODS:
                        nre = 0
                        while nre < NUM_REP:

                            string_exp = sea + '_leg_' + str(leg) + '_' + sep_method + '_' + meth + '_' +  varset + '_' + str(nre)
                            nre += 1

                            data_f = data.copy()
                            if 'leg' not in data.columns.tolist():
                                data_f = dataset.add_legs_index(data_f)

                            data_f = data_f.loc[data_f['leg'] == leg]
                            data_f.drop('leg', axis=1, inplace=True)

                            leg_whole_ = dataset.subset_data_stack_variables(
                            data_f, # 
                            varset=varset.lower(), # 
                            seatype=sea.lower(),
                            )

                            leg_whole_ = leg_whole_.dropna()
                            s1, s2 = leg_whole_.shape

                            if s1 < 10:
                                continue

                            if not TRN_TEST_INDEX.values.any():

                                separation = SPLIT_SIZE
                                trn_size = ceil(s1*separation) #;

                                if sep_method.lower() == 'prediction':
                                #     print('training data until ' + str(separation) + ', then test.')
                                    trn = leg_whole_.iloc[:trn_size,:].copy()
                                    tst = leg_whole_.iloc[trn_size:,:].copy()

                                elif sep_method.lower() == 'interpolation':
                                #     print('training data random %f pc subset, rest test'%(separation*100))
                                    leg_whole_ = shuffle(leg_whole_)
                                    trn = leg_whole_.iloc[:trn_size,:].copy()
                                    tst = leg_whole_.iloc[trn_size:,:].copy()

                            elif TRN_TEST_INDEX.values.any():

                                trn = leg_whole_.loc[TRN_TEST_INDEX.values  == 1,:].copy()
                                tst = leg_whole_.loc[TRN_TEST_INDEX.values == 2,:].copy()

                            # Standardize data to 0 mean unit variance based on training statistics (assuming stationarity)
                            inds_trn = trn.index

                            # SCALE TRAINING DATA X, y
                            if NORMALIZE_X:
                                scalerX = preprocessing.StandardScaler().fit(trn.iloc[:,:-1])
                                X = scalerX.transform(trn.iloc[:,:-1])#, columns=trn.iloc[:,:-1].columns, index=trn.index)
                            else:
                                X = trn.iloc[:,:-1]

                            if NORMALIZE_Y:
                                scalerY = preprocessing.StandardScaler().fit(trn.iloc[:,-1].values.reshape(-1, 1))
                                y = scalerY.transform(trn.iloc[:,-1].values.reshape(-1, 1))
                            else:
                                y = trn.iloc[:,-1]

                            del trn
                            # ------------
                            ######### 1 : Ridge Regression
                            if meth.lower() == 'ridgereg':
                                MSE_error = make_scorer(mean_squared_error, greater_is_better=False)
                                regModel = RidgeCV(alphas=np.logspace(-3,3,7), fit_intercept=not NORMALIZE_Y,
                                    normalize=False, store_cv_values=False, gcv_mode='svd',
                                    cv=5, scoring=MSE_error).fit(X,y) #(trn.iloc[:,:-1], trn.iloc[:,-1])
                                regModel.coef_ = regModel.coef_[0]

                            elif meth.lower() == 'pls':
                                n = 3
                                regModel = PLSRegression(n_components=n, scale=False).fit(X,y)
                                regModel.coef_ = np.squeeze(np.transpose(regModel.coef_))

                            elif meth.lower() == 'lasso':
                                regModel = LassoCV(alphas=np.logspace(-3,-1,3), n_alphas=200,
                                                   fit_intercept=not NORMALIZE_Y, cv=5).fit(X,y)

                            elif meth.lower() == 'lingpr':
                                kernel = kernels.DotProduct(sigma_0=1,  sigma_0_bounds=(1e-05, 1e05)) + \
                                     1.0 * kernels.WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-3, 1e+3))
                                regModel = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b',
                                                                    alpha=0, n_restarts_optimizer=5).fit(X,y)
                            elif meth.lower() == 'rf':
                                import sklearn.ensemble
                                regModel = sklearn.ensemble.RandomForestRegressor(n_estimators=100,
                                        criterion='mse', max_features=0.5,
                                        max_depth=20, min_samples_split=2,
                                        min_samples_leaf=1, ).fit(X,np.ravel(y))
                                regModel.coef_ = regModel.feature_importances_

                            elif meth.lower() == 'rbfgpr':
                                kernel = 1.0 * kern.RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + \
                                1.0 * kern.WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-1, 1e+4)) + \
                                1.0 * kern.ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-05, 100000.0)) + \
                                1.0 * kern.DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-05, 100000.0))

                                regModel = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b',
                                                                    alpha=0,
                                                                    n_restarts_optimizer=5).fit(X,y)
                            elif meth.lower() == 'rbfgprard':

                                #x = trn.iloc[:,:-1].values
                                #y = trn.iloc[:,-1].values.reshape(-1,1)
                                s1 = X.shape[1]

                                k = (GPy.kern.RBF(s1, ARD=True)
                                     + GPy.kern.White(s1, 0.001)
                                     + GPy.kern.Bias(s1, 0.001))
                                     #+ GPy.kern.Linear(s1, variances=0.001, ARD=False))

                                regModel = GPy.models.GPRegression(X, y, kernel=k)
                                #regModel.optimize_restarts(parallel=True, robust=True, num_restarts=5, max_iters=200)
                                regModel.optimize('scg', max_iters=200) # 'scg'
                                regModel.coef_ = np.array(regModel.sum.rbf.lengthscale)

                            else:
                                print('method not implemented yet. Or check the spelling')
                                break

                            inds_tst = tst.index

                            if NORMALIZE_X:
                                x = scalerX.transform(tst.iloc[:,:-1])#, columns=tst.iloc[:,:-1].columns, index=tst.index)
                            else:
                                x = tst.iloc[:,:-1]

                            y_ts_gt = tst.iloc[:,-1]
                            del tst

                            if meth.lower() == 'rbfgprard':
                                # x_ = tst_.values
                                # x  = trn.iloc[:,:-1].values
                                y_ts_h = regModel.predict(x)[0].reshape(-1,)
                                y_ts_h = pd.Series(y_ts_h,index=inds_tst)
                                y_tr_h = regModel.predict(X)[0].reshape(-1,)
                                y_tr_h = pd.Series(y_tr_h,index=inds_trn)
                            else:
                                y_ts_h = regModel.predict(x).reshape(-1,)
                                y_ts_h = pd.Series(y_ts_h,index=inds_tst)
                                y_tr_h = regModel.predict(X).reshape(-1,)
                                y_tr_h = pd.Series(y_tr_h,index=inds_trn)


                            if NORMALIZE_Y:
                                y_tr_h = scalerY.inverse_transform(y_tr_h)
                                y_ts_h = scalerY.inverse_transform(y_ts_h)
                                y_tr_gt = scalerY.inverse_transform(y)#trn.iloc[:,-1]
                            else:
                                y_tr_gt = y#trn.iloc[:,-1]


                            # Compute scores
                            mse = np.sqrt(mean_squared_error(y_ts_gt, y_ts_h))
                            r2 = r2_score(y_ts_gt, y_ts_h)

                            t_mse = np.sqrt(mean_squared_error(y_tr_gt, y_tr_h))
                            t_r2 = r2_score(y_tr_gt, y_tr_h)

                            if hasattr(regModel, 'alpha_') & hasattr(regModel, 'coef_'):
                                summ[string_exp] = {'regularizer': regModel.alpha_,
                                                    'weights': regModel.coef_,
                                                    'tr_RMSE': t_mse,
                                                    'tr_R2': t_r2,
                                                    'ts_RMSE': mse,
                                                    'ts_R2': r2}#,
                                                    # 'y_tr_hat': y_tr_h,
                                                    # 'y_ts_hat': y_ts_h}

                            elif hasattr(regModel, 'coef_') & ~hasattr(regModel, 'alpha_'):
                                summ[string_exp] = {'weights': regModel.coef_,
                                                    'tr_RMSE': t_mse,
                                                    'tr_R2': t_r2,
                                                    'ts_RMSE': mse,
                                                    'ts_R2': r2}#,
                                                    # 'y_tr_hat': y_tr_h,
                                                    # 'y_ts_hat': y_ts_h}
                            else:
                                summ[string_exp] = {'tr_RMSE': t_mse,
                                                    'tr_R2': t_r2,
                                                    'ts_RMSE': mse,
                                                    'ts_R2': r2}#,
                                                    # 'y_tr_hat': y_tr_h,
                                                    # 'y_ts_hat': y_ts_h}


                            #if SAVE_TEXT_DUMP:
                                # results = pd.DataFrame(index=[],columns=['n_ts', 'tst_r2','tst_rmse',' n_tr', 'trn_r2','trn_rmse'])
                            #    results.loc[nre-1] = [len(y_ts_h), r2, mse, len(y_tr_h), t_r2, t_mse]

                            del leg_whole_, regModel, y_tr_gt, y_ts_gt, y_tr_h, y_ts_h

    save_obj(summ, SAVEFOLDER / MODELNAME)
    #results.to_csv(path_or_buf=SAVEFOLDER + MODELNAME + '.csv', sep='\t')
    return summ

##############################################################################################################
def run_baselines_particle_size(data, options):
    """
        Run regression model(s) on single replica of data (no random resampling, but uses indexes). Good for testing training and testing on manually specified splits

        :param data: input dataset, training and test mixed, [x,y] labels last column
        :param kwargs: Dictionary containing options.  Fields are:

        |    SEP_METHOD : ['interpolation','prediction'] -> Mode of learning / testing
        |    NUM_REP : N -> N random resampling of training and test sets (ONLY 'interpolation')
        |    LEG_P : [1, 2, 3] -> On which leg to predict. Treated separately by the models
        |    METHODS : [ridgereg', 'pls', 'lasso', 'rbfgpr', 'rbfgprard', 'rf'] -> Regression method
        |    NORMALIZE_Y : BOOL -> whether to normalize outputs y
        |    NORMALIZE_X : BOOL -> whether to normalize inputs x
        |    SAVEFOLDER : STRING -> folder address to store results
        |    MODELNAME : STRING -> name of the model file and scores
        |    SPLIT_SIZE : FLOAT [0,1] -> percentage of training to test datapoints
        |    TRN_TEST_INDEX : DF -> list of integers containing wheter the point belongs to training (=1) or to the test set (=2). Has to be same size of data.shape[0]

        :returns: A dictionary containing weights, accuracy scores for training and test sets
    """

    SEP_METHOD = options['SEP_METHOD']
    NUM_REP = options['NUM_REP']
    LEG_P = options['LEG_P']
    METHODS = options['METHODS']
    NORMALIZE_Y = options['NORMALIZE_Y']
    NORMALIZE_X = options['NORMALIZE_X']
    SAVEFOLDER = options['SAVEFOLDER']
    MODELNAME = options['MODELNAME']
    SPLIT_SIZE = options['SPLIT_SIZE']
    TRN_TEST_INDEX = options['TRN_TEST_INDEX']

    #SAVE_TEXT_DUMP = kwargs['SAVE_TEXT_DUMP']
    if not os.path.isdir(SAVEFOLDER):
        os.mkdir(SAVEFOLDER)

    if os.path.exists(SAVEFOLDER / MODELNAME):
        print("file exists, overwriting")

    summ = {}
    #if SAVE_TEXT_DUMP:
    #    results = pd.DataFrame(index=[],columns=['tst_r2','tst_rmse','trn_r2','trn_rmse','n_tr','n_ts'])

    for sep_method in SEP_METHOD:
        #print(sep_method)
        for leg in LEG_P:
            # print(leg)
            for meth in METHODS:
                nre = 0
                while nre < NUM_REP:

                    string_exp = 'leg_' + str(leg) + '_' + sep_method + '_' + meth + '_' + str(nre)
                    nre += 1

                    data_f = data.copy()

                    if 'leg' not in data.columns.tolist():
                        data_f = dataset.add_legs_index(data_f)

                    data_f = data_f.loc[data_f['leg'] == leg]
                    data_f.drop('leg', axis=1, inplace=True)

                    leg_whole_ = data_f.dropna()
                    s1, s2 = leg_whole_.shape

                    if s1 < 10:
                        continue

                    if not TRN_TEST_INDEX.values.any():

                        separation = SPLIT_SIZE
                        trn_size = ceil(s1*separation) #;

                        if sep_method.lower() == 'prediction':
                        #     print('training data until ' + str(separation) + ', then test.')
                            trn = leg_whole_.iloc[:trn_size,:].copy()
                            tst = leg_whole_.iloc[trn_size:,:].copy()

                        elif sep_method.lower() == 'interpolation':
                        #     print('training data random %f pc subset, rest test'%(separation*100))
                            leg_whole_ = shuffle(leg_whole_)
                            trn = leg_whole_.iloc[:trn_size,:].copy()
                            tst = leg_whole_.iloc[trn_size:,:].copy()

                    elif TRN_TEST_INDEX.values.any():

                        trn = leg_whole_.loc[TRN_TEST_INDEX.values  == 1,:].copy()
                        tst = leg_whole_.loc[TRN_TEST_INDEX.values == 2,:].copy()

                    # Standardize data to 0 mean unit variance based on training statistics (assuming stationarity)
                    inds_trn = trn.index

                    # SCALE TRAINING DATA X, y
                    if NORMALIZE_X:
                        scalerX = preprocessing.StandardScaler().fit(trn.iloc[:,:-1])
                        X = scalerX.transform(trn.iloc[:,:-1])#, columns=trn.iloc[:,:-1].columns, index=trn.index)
                    else:
                        X = trn.iloc[:,:-1]

                    if NORMALIZE_Y:
                        scalerY = preprocessing.StandardScaler().fit(trn.iloc[:,-1].values.reshape(-1, 1))
                        y = scalerY.transform(trn.iloc[:,-1].values.reshape(-1, 1))
                    else:
                        y = trn.iloc[:,-1]

                    del trn
                    # 

                    ######### 1 : Ridge Regression
                    if meth.lower() == 'ridgereg':
                        MSE_error = make_scorer(mean_squared_error, greater_is_better=False)
                        regModel = RidgeCV(alphas=np.logspace(-3,0), fit_intercept=True,
                            normalize=False, store_cv_values=False, gcv_mode='svd',
                            cv=5).fit(X,y) #(trn.iloc[:,:-1], trn.iloc[:,-1]
                        regModel.coef_ = regModel.coef_[0]

                    elif meth.lower() == 'bayesianreg':
                        regModel = sk.linear_model.BayesianRidge(n_iter=300, tol=1.e-6, alpha_1=1.e-6, alpha_2=1.e-6, lambda_1=1.e-6, lambda_2=1.e-6, compute_score=True, fit_intercept=True, normalize=False).fit(X,y.ravel())


                    elif meth.lower() == 'pls':
                        n = 3
                        regModel = PLSRegression(n_components=n, scale=False).fit(X,y)
                        regModel.coef_ = np.squeeze(np.transpose(regModel.coef_))

                    elif meth.lower() == 'lasso':
                        regModel = LassoCV(alphas=np.logspace(-2,0,1), n_alphas=500,
                                           fit_intercept=True, max_iter=5000, cv=5).fit(X,y.ravel())
                    elif meth.lower() == 'lingpr':
                        kernel = kernels.DotProduct(sigma_0 = 1, sigma_0_bounds=(1e-05, 1e05)) + \
                             1.0 * kernels.WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-3, 1e+3))
                        regModel = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b',
                                                            alpha=0, n_restarts_optimizer=5).fit(X,y)
                        print(kernel)
                    elif meth.lower() == 'rf':
                        import sklearn.ensemble
                        regModel = sklearn.ensemble.RandomForestRegressor(n_estimators=100,
                                criterion='mse', max_features=0.5,
                                max_depth=20, min_samples_split=2,
                                min_samples_leaf=1, ).fit(X,np.ravel(y))
                        regModel.coef_ = regModel.feature_importances_

                    elif meth.lower() == 'rbfgpr':
                        kernel = 1.0 * kern.RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + \
                        1.0 * kern.WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-1, 1e+4)) + \
                        1.0 * kern.ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-05, 100000.0)) + \
                        1.0 * kern.DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-05, 100000.0))

                        regModel = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b',
                                                            alpha=0,
                                                            n_restarts_optimizer=5).fit(X,y)
                    elif meth.lower() == 'rbfgprard':

                        #x = trn.iloc[:,:-1].values
                        #y = trn.iloc[:,-1].values.reshape(-1,1)
                        s1 = X.shape[1]

                        k = (GPy.kern.RBF(s1, ARD=True)
                             + GPy.kern.White(s1, 0.001)
                             + GPy.kern.Bias(s1, 0.001))
                             #+ GPy.kern.Linear(s1, variances=0.001, ARD=False))

                        regModel = GPy.models.GPRegression(X, y, kernel=k)
                        #regModel.optimize_restarts(parallel=True, robust=True, num_restarts=5, max_iters=200)
                        regModel.optimize('scg', max_iters=200) # 'scg'
                        regModel.coef_ = np.array(regModel.sum.rbf.lengthscale)

                    else:
                        print('method not implemented yet. Or check the spelling')
                        break

                    inds_tst = tst.index

                    if NORMALIZE_X:
                        x = scalerX.transform(tst.iloc[:,:-1])#, columns=tst.iloc[:,:-1].columns, index=tst.index)
                    else:
                        x = tst.iloc[:,:-1]

                    y_ts_gt = tst.iloc[:,-1]
                    del tst

                    if meth.lower() == 'rbfgprard':
                        # x_ = tst_.values
                        # x  = trn.iloc[:,:-1].values
                        y_ts_h = regModel.predict(x)[0].reshape(-1,)
                        y_ts_h = pd.Series(y_ts_h,index=inds_tst)
                        y_tr_h = regModel.predict(X)[0].reshape(-1,)
                        y_tr_h = pd.Series(y_tr_h,index=inds_trn)
                    else:
                        y_ts_h = regModel.predict(x).reshape(-1,)
                        y_ts_h = pd.Series(y_ts_h,index=inds_tst)
                        y_tr_h = regModel.predict(X).reshape(-1,)
                        y_tr_h = pd.Series(y_tr_h,index=inds_trn)


                    if NORMALIZE_Y:
                        y_tr_h = scalerY.inverse_transform(y_tr_h)
                        y_ts_h = scalerY.inverse_transform(y_ts_h)
                        y_tr_gt = scalerY.inverse_transform(y)#trn.iloc[:,-1]
                    else:
                        y_tr_gt = y#trn.iloc[:,-1]


                    # Compute scores
                    mse = np.sqrt(mean_squared_error(y_ts_gt, y_ts_h))
                    r2 = r2_score(y_ts_gt, y_ts_h)

                    t_mse = np.sqrt(mean_squared_error(y_tr_gt, y_tr_h))
                    t_r2 = r2_score(y_tr_gt, y_tr_h)

                    if hasattr(regModel, 'alpha_') & hasattr(regModel, 'coef_'):
                        summ[string_exp] = {'regularizer': regModel.alpha_,
                                            'weights': regModel.coef_,
                                            'tr_RMSE': t_mse,
                                            'tr_R2': t_r2,
                                            'ts_RMSE': mse,
                                            'ts_R2': r2}#,
                                            # 'y_tr_hat': y_tr_h,
                                            # 'y_ts_hat': y_ts_h}

                    elif hasattr(regModel, 'coef_') & ~hasattr(regModel, 'alpha_'):
                        summ[string_exp] = {'weights': regModel.coef_,
                                            'tr_RMSE': t_mse,
                                            'tr_R2': t_r2,
                                            'ts_RMSE': mse,
                                            'ts_R2': r2}#,
                                            # 'y_tr_hat': y_tr_h,
                                            # 'y_ts_hat': y_ts_h}
                    else:
                        summ[string_exp] = {'tr_RMSE': t_mse,
                                            'tr_R2': t_r2,
                                            'ts_RMSE': mse,
                                            'ts_R2': r2}#,
                                            # 'y_tr_hat': y_tr_h,
                                            # 'y_ts_hat': y_ts_h}


                    #if SAVE_TEXT_DUMP:
                        # results = pd.DataFrame(index=[],columns=['n_ts', 'tst_r2','tst_rmse',' n_tr', 'trn_r2','trn_rmse'])
                    #    results.loc[nre-1] = [len(y_ts_h), r2, mse, len(y_tr_h), t_r2, t_mse]

                    del leg_whole_, regModel, y_tr_gt, y_ts_gt, y_tr_h, y_ts_h

    save_obj(summ, SAVEFOLDER / MODELNAME)
    #results.to_csv(path_or_buf=SAVEFOLDER + MODELNAME + '.csv', sep='\t')
    return summ

##############################################################################################################
def run_regression_indexed_data(data, inds, regression_model, NORM_X=True, NORM_Y=True):
    """
        Run regression model(s) on single replica of data (no random resampling). Good for testing training and testing on manually specified splits

        :param data: df, input dataset, training and test mixed, [x,y] labels last column
        :param inds: df or series, index vector of training (ind = 1) and test (ind = 2,...,S) for S test SPLITS
        :param regression_model: list of stings, regression method. Options are hard coded here, but can be extracted in a dict in the future
        :param NORM_X: bool, wheter to normalize input data
        :param NORM_Y: bool, wheter to normalize output data
        :returns: dict containing weights, accuracy scores for training and tests, and the time difference between first and last training points
    """

    tr_ = data.loc[inds.loc[inds['ind'] == 1].index,:].copy()

    if NORM_X:
        scalerX = sk.preprocessing.StandardScaler().fit(tr_.iloc[:,:-1])
        trn = pd.DataFrame(scalerX.transform(tr_.iloc[:,:-1]), columns=tr_.iloc[:,:-1].columns,
                            index=tr_.index)
    else:
        trn = tr_.iloc[:,:-1]

    if NORM_Y:
        scalerY = sk.preprocessing.StandardScaler().fit(tr_.iloc[:,-1].values.reshape(-1, 1))
        y_trn = scalerY.transform(tr_.iloc[:,-1].values.reshape(-1, 1))
    else:
        y_trn = tr_.iloc[:,-1]

    trn = trn.assign(labels=y_trn)
    #             print(trn.columns.tolist())


    if regression_model.lower() == 'ridgereg':
    #                 MSE_error = make_scorer(mean_squared_error, greater_is_better=False)
    #                 regModel = RidgeCV(alphas=np.logspace(-6,6,13), fit_intercept=not NORM_Y,
    #                            normalize=False, store_cv_values=False, gcv_mode='svd',
    #                            cv=3, scoring=MSE_error).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
        regModel = sk.linear_model.Ridge(alpha=0.1, fit_intercept=not NORM_Y,
                   normalize=False).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
        weights = regModel.coef_

    elif regression_model.lower() == 'lasso':
    #                 regModel = LassoCV(alphas=np.logspace(-3,-1,3), n_alphas=200,
    #                                     fit_intercept=not NORM_Y, cv=3).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
        regModel = sk.linear_model.Lasso(alpha=0.1, fit_intercept=not NORM_Y,
                    normalize=False).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
        weights = regModel.coef_

    elif regression_model.lower() == 'pls':
        n = 3
        regModel = PLSRegression(n_components=n, scale=False).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
        regModel.coef_ = np.squeeze(np.transpose(regModel.coef_))
        weights = regModel.coef_

    elif regression_model.lower() == 'rf':
        import sklearn.ensemble
        regModel = sklearn.ensemble.RandomForestRegressor(n_estimators=100, criterion='mse',
                max_depth=10, min_samples_split=2, min_samples_leaf=1).fit(trn.iloc[:,:-1], trn.iloc[:,-1])

    elif regression_model.lower() == 'rbfgpr':
        kernel = 1.0 * kern.RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + \
        1.0 * kern.WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-1, 1e+4)) + \
        1.0 * kern.ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-05, 100000.0)) + \
        1.0 * kern.DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-05, 100000.0))

        regModel = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b',
                        alpha=0, n_restarts_optimizer=5).fit(trn.iloc[:,:-1], trn.iloc[:,-1])

    elif regression_model.lower() == 'rbfgprard':

        inds_trn = trn.index
        x = trn.iloc[:,:-1].values
        y = trn.iloc[:,-1].values.reshape(-1,1)

        k = (GPy.kern.RBF(x.shape[1], ARD=True)
             + GPy.kern.White(x.shape[1], 0.01)
             + GPy.kern.Linear(x.shape[1], variances=0.01, ARD=False))

        regModel = GPy.models.GPRegression(x,y,kernel=k)
        regModel.optimize('bfgs', max_iters=200)
    #                 print(regModel)
        weights = 50/regModel.sum.rbf.lengthscale

    else:
        print('method not implemented yet. Or check the spelling')
        return []

    #             from_to = [str(trn.index.tolist()[0]) + ' - ' + str(trn.index.tolist()[-1])]
    gap_time_delta = str(trn.index.tolist()[-1] - trn.index.tolist()[0])

    # weights_summary[gap_num,:] =

    tr_r2 = []; tr_mse = []# [[],[]]; mse = [[],[]]
    ts_r2 = []; ts_mse = []

    a = 1
    for bb in np.setdiff1d(np.unique(inds),1):

        ts_ = data.loc[inds.loc[inds['ind'] == bb].index,:]

        if NORM_X:
            tst = pd.DataFrame(scalerX.transform(ts_.iloc[:,:-1]), columns=ts_.iloc[:,:-1].columns, index=ts_.index)
        else:
            tst = ts_.iloc[:,:-1]

        if NORM_Y:
            y_tst = scalerY.transform(ts_.iloc[:,-1].values.reshape(-1, 1))
        else:
            y_tst = ts_.iloc[:,-1]

        tst = tst.assign(labels=y_tst)

        if regression_model.lower() == 'rbfgprard':
            inds_tst = tst.index
            x_ = tst.iloc[:,:-1].values
            y_ts_h = regModel.predict(x_)[0].reshape(-1,)
            y_ts_h = pd.Series(y_ts_h,index=inds_tst)
            y_tr_h = regModel.predict(x)[0].reshape(-1,)
            y_tr_h = pd.Series(y_tr_h,index=inds_trn)
        else:
            y_ts_h = regModel.predict(tst.iloc[:,:-1])
            y_tr_h = regModel.predict(trn.iloc[:,:-1])


        if NORM_Y:
            y_tr_h = scalerY.inverse_transform(y_tr_h)
            y_ts_h = scalerY.inverse_transform(y_ts_h)
            y_tr_gt = scalerY.inverse_transform(trn.iloc[:,-1])
            y_ts_gt = scalerY.inverse_transform(tst.iloc[:,-1])
        else:
            y_tr_gt = trn.iloc[:,-1]
            y_ts_gt = tst.iloc[:,-1]
        if a == 1:
            tr_r2.append(r2_score(y_tr_gt, y_tr_h))
            tr_mse.append(np.sqrt(mean_squared_error(y_tr_gt, y_tr_h)))
            a = 2
        ts_r2.append(r2_score(y_ts_gt, y_ts_h))
        ts_mse.append(np.sqrt(mean_squared_error(y_ts_gt, y_ts_h)))

        if 0:
            print('trn: MSE %f, R2 %f' %(t_mse,t_r2))
            print('%f -- trn: MSE %f, R2 %f' %(bb,t_mse,t_r2))
            print('%f -- tst: MSE %f, R2 %f' %(bb,mse,r2))

    del inds

    return {'weights': weights, 'gap_time_delta': gap_time_delta, 'tr_r2': tr_r2,
                'ts_r2': ts_r2, 'tr_mse': tr_mse, 'ts_mse': ts_mse}

##############################################################################################################
def run_regression_simple_data(data_tr, data_ts, regression_model, NORM_X=True, NORM_Y=True):
    """
        Run regression model(s) on single replica of data

        :param data_tr: df, input training dataset, [x,y] labels last column
        :param data_ts: df, input test dataset, [x,y] labels last column
        :param regression_model: list of stings, regression method. Options are hard coded here, but can be extracted in a dict in the future
        :param NORM_X: bool, wheter to normalize input data
        :param NORM_Y: bool, wheter to normalize output data
        :returns: dict containing weights, accuracy scores for training and tests, and the time difference between first and last training points

    """

    tr_ = data_tr.copy()
    ts_ = data_ts.copy()

    if NORM_X:
        scalerX = sk.preprocessing.StandardScaler().fit(tr_.iloc[:,:-1])
        trn = pd.DataFrame(scalerX.transform(tr_.iloc[:,:-1]), columns=tr_.iloc[:,:-1].columns,
                            index=tr_.index)
        tst = pd.DataFrame(scalerX.transform(ts_.iloc[:,:-1]), columns=ts_.iloc[:,:-1].columns, index=ts_.index)
    else:
        trn = tr_.iloc[:,:-1]
        tst = ts_.iloc[:,:-1]

    if NORM_Y:
        scalerY = sk.preprocessing.StandardScaler().fit(tr_.iloc[:,-1].values.reshape(-1, 1))
        y_trn = scalerY.transform(tr_.iloc[:,-1].values.reshape(-1, 1))
        y_tst = scalerY.transform(ts_.iloc[:,-1].values.reshape(-1, 1))
    else:
        y_trn = tr_.iloc[:,-1]
        y_tst = ts_.iloc[:,-1]

    trn = trn.assign(labels=y_trn)
    tst = tst.assign(labels=y_tst)

    if regression_model.lower() == 'ridgereg':
    #                 MSE_error = make_scorer(mean_squared_error, greater_is_better=False)
    #                 regModel = RidgeCV(alphas=np.logspace(-6,6,13), fit_intercept=not NORM_Y,
    #                            normalize=False, store_cv_values=False, gcv_mode='svd',
    #                            cv=3, scoring=MSE_error).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
        regModel = sk.linear_model.Ridge(alpha=0.1, fit_intercept=not NORM_Y,
                   normalize=False).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
        weights = regModel.coef_

    elif regression_model.lower() == 'lasso':
    #                 regModel = LassoCV(alphas=np.logspace(-3,-1,3), n_alphas=200,
    #                                     fit_intercept=not NORM_Y, cv=3).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
        regModel = sk.linear_model.Lasso(alpha=0.1, fit_intercept=not NORM_Y,
                    normalize=False).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
        weights = regModel.coef_

    elif regression_model.lower() == 'pls':
        n = 3
        regModel = PLSRegression(n_components=n, scale=False).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
        regModel.coef_ = np.squeeze(np.transpose(regModel.coef_))
        weights = regModel.coef_

    elif regression_model.lower() == 'rf':
        import sklearn.ensemble
        regModel = sklearn.ensemble.RandomForestRegressor(n_estimators=100, criterion='mse',
                max_features = 0.5, max_depth=20, min_samples_split=2,
                min_samples_leaf=1).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
        weights = regModel.feature_importances_

    elif regression_model.lower() == 'rbfgpr':
        kernel = 1.0 * kern.RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + \
        1.0 * kern.WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-1, 1e+4)) + \
        1.0 * kern.ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-05, 100000.0)) + \
        1.0 * kern.DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-05, 100000.0))

        regModel = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b',
                        alpha=0, n_restarts_optimizer=5).fit(trn.iloc[:,:-1], trn.iloc[:,-1])

    elif regression_model.lower() == 'rbfgprard':

        inds_trn = trn.index
        x = trn.iloc[:,:-1].values
        y = trn.iloc[:,-1].values.reshape(-1,1)

        k = (GPy.kern.RBF(x.shape[1], ARD=True)
             + GPy.kern.White(x.shape[1], 0.01)
             + GPy.kern.Linear(x.shape[1], variances=0.01, ARD=False))

        regModel = GPy.models.GPRegression(x,y,kernel=k)
        regModel.optimize('bfgs', max_iters=200)
    #                 print(regModel)
        weights = 50/regModel.sum.rbf.lengthscale

    else:
        print('method not implemented yet. Or check the spelling')
        return []

    # TEST STEPS
    if regression_model.lower() == 'rbfgprard':
        inds_tst = tst.index
        x_ = tst.iloc[:,:-1].values
        y_ts_h = regModel.predict(x_)[0].reshape(-1,)
        y_ts_h = pd.Series(y_ts_h,index=inds_tst)
        y_tr_h = regModel.predict(x)[0].reshape(-1,)
        y_tr_h = pd.Series(y_tr_h,index=inds_trn)
    else:
        y_ts_h = regModel.predict(tst.iloc[:,:-1])
        y_tr_h = regModel.predict(trn.iloc[:,:-1])

    if NORM_Y:
        y_tr_h = scalerY.inverse_transform(y_tr_h)
        y_ts_h = scalerY.inverse_transform(y_ts_h)
        y_tr_gt = scalerY.inverse_transform(trn.iloc[:,-1])
        y_ts_gt = scalerY.inverse_transform(tst.iloc[:,-1])
    else:
        y_tr_gt = trn.iloc[:,-1]
        y_ts_gt = tst.iloc[:,-1]

    tr_r2 = r2_score(y_tr_gt, y_tr_h)
    tr_mse = np.sqrt(mean_squared_error(y_tr_gt, y_tr_h))
    ts_r2 = r2_score(y_ts_gt, y_ts_h)
    ts_mse = np.sqrt(mean_squared_error(y_ts_gt, y_ts_h))

    if 0:
        print('trn: MSE %f, R2 %f' %(t_mse,t_r2))
        print('%f -- trn: MSE %f, R2 %f' %(bb,t_mse,t_r2))
        print('%f -- tst: MSE %f, R2 %f' %(bb,mse,r2))

    return {'weights': weights, 'tr_r2': tr_r2,
                'ts_r2': ts_r2, 'tr_mse': tr_mse, 'ts_mse': ts_mse}
