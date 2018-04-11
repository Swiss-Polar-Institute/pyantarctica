import pandas as pd
import numpy as np
import pickle
import os

from datetime import datetime 
from math import ceil

import ace_package.dataset as dataset 
from ace_package.modeling import plot_predicted_timeseries

import sklearn as sk
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel
from sklearn.model_selection import GridSearchCV

def save_obj(obj, fname):
    with open(fname + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def run_baselines_num_conc_aerosol(data, **kwargs):
    
    SEP_METHOD = kwargs['SEP_METHOD']
    SEA = kwargs['SEA']
    NUM_REP = kwargs['NUM_REP']
    LEG_P = kwargs['LEG_P']
    VARSET = kwargs['VARSET']
    METHODS = kwargs['METHODS']
    NORMALIZE_Y = kwargs['NORMALIZE_Y']
    SAVEFOLDER = kwargs['SAVEFOLDER']
    MODELNAME = kwargs['MODELNAME']
    
    if not os.path.isdir(SAVEFOLDER):
        os.mkdir(SAVEFOLDER)

    if os.path.exists(SAVEFOLDER + MODELNAME): 
        print("file exists, overwriting")
    
    summ = {}
    
    for sep_method in SEP_METHOD: 
        print(sep_method)
        for leg in LEG_P: 
            print(leg)
            for sea in SEA: 
                for varset in VARSET:
                    for meth in METHODS:
                        nre = 0
                        while nre < NUM_REP:

                            string_exp = sea + '_leg_' + str(leg) + '_' + sep_method + '_' + meth + '_' +  varset + '_' + str(nre)
                            # print(string_exp)
                            nre += 1 

                            # if (varset.lower() == 'full' and leg != 1):
                            #     continue


                            if varset.lower() == 'full':
                                cols_total = ['hs', 'tp', 'steep', 'phase_vel', 'age', 'wind', 'num_conc']
                                cols_wind  = ['hs_w', 'tp_w', 'steep_w', 'phase_vel_w', 'age_w', 'wind', 'num_conc']
                            elif varset.lower() == 'nowind': 
                                cols_total = ['hs', 'tp', 'steep', 'phase_vel', 'num_conc']
                                cols_wind  = ['hs_w', 'tp_w', 'steep_w', 'phase_vel_w', 'num_conc']
                            elif varset.lower() == 'reduced':
                                cols_total = ['hs', 'tp', 'wind', 'num_conc']
                                cols_wind  = ['hs_w', 'tp_w', 'wind', 'num_conc']

                            if sea.lower() == 'total':
                                leg_whole_ = data.loc[data['leg'] == leg, cols_total].dropna().copy()
                            
                            elif sea.lower() == 'wind':
                                leg_whole_ = data.loc[data['leg'] == leg, cols_wind].dropna().copy()
                            
                            s1, s2 = leg_whole_.shape

                            separation=2.0/3    
                            trn_size = ceil(s1*separation) #; print(trn_size, s1)

                            if sep_method.lower() == 'prediction':
                            #     print('training data until ' + str(separation) + ', then test.')
                                trn = leg_whole_.iloc[:trn_size,:].copy()
                                tst = leg_whole_.iloc[trn_size:,:].copy()

                            elif sep_method.lower() == 'interpolation': 
                            #     print('training data random %f pc subset, rest test'%(separation*100))
                                leg_whole_ = shuffle(leg_whole_)
                                trn = leg_whole_.iloc[:trn_size,:].copy()
                                tst = leg_whole_.iloc[trn_size:,:].copy()
                            
                            # Standardize data to 0 mean unit variance based on training statistics (stationarity)
                            # ----------
                            scalerX = preprocessing.StandardScaler().fit(trn.iloc[:,:-1])

                            if NORMALIZE_Y: 
                                scalerY = preprocessing.StandardScaler().fit(trn.iloc[:,-1].values.reshape(-1, 1))

                            trn_ = pd.DataFrame(scalerX.transform(trn.iloc[:,:-1]), columns=trn.iloc[:,:-1].columns, index=trn.index)
                            tst_ = pd.DataFrame(scalerX.transform(tst.iloc[:,:-1]), columns=tst.iloc[:,:-1].columns, index=tst.index)

                            if NORMALIZE_Y: 
                                scalerY = preprocessing.StandardScaler().fit(trn.iloc[:,-1].values.reshape(-1, 1))
                                y_trn = scalerY.transform(trn.iloc[:,-1].values.reshape(-1, 1))
                                y_tst = scalerY.transform(tst.iloc[:,-1].values.reshape(-1, 1))
                            else: 
                                y_trn = trn.iloc[:,-1]
                                y_tst = tst.iloc[:,-1]

                            trn = trn_; trn['num_conc'] = y_trn
                            tst = tst_; tst['num_conc'] = y_tst

                            # y_gt_scaled = pd.DataFrame(scaler.transform(leg_whole_), columns=leg_whole_.columns, index=leg_whole_.index)
                            # y_gt_scaled = y_gt_scaled.iloc[:,-1]
                            # ------------
                            ######### 1 : Ridge Regression
                            if meth.lower() == 'ridgereg':
                                MSE_error = make_scorer(mean_squared_error, greater_is_better=False)
                                regModel = RidgeCV(alphas=np.logspace(-6,6,13), fit_intercept=not NORMALIZE_Y, normalize=False, store_cv_values=False, gcv_mode='svd', cv=5, scoring=MSE_error)#.fit(trn.iloc[:,:-1], trn.iloc[:,-1])
                            elif meth.lower() == 'pls':
                                n = 3
                                regModel = PLSRegression(n_components=n, scale=False)#.fit(trn.iloc[:,:-1], trn.iloc[:,-1])

                            elif meth.lower() == 'lasso':
                                regModel = LassoCV(alphas=np.logspace(-3,-1,3), n_alphas=200, fit_intercept=not NORMALIZE_Y, cv=5)#.fit(trn.iloc[:,:-1], trn.iloc[:,-1])

                            elif meth.lower() == 'lingpr':
                                kernel = DotProduct(sigma_0=1,  sigma_0_bounds=(1e-05, 1e05)) + \
                                     1.0 * WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-3, 1e+3))
                                regModel = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', alpha=0, n_restarts_optimizer=5)
                                #parameters = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'sigma_0':[0.01, 1, 10]}

                            elif meth.lower() == 'rbfgpr':
                                kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + \
                                     1.0 * WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-3, 1e+3))
                                regModel = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', alpha=0, n_restarts_optimizer=5)
                            else: 
                                print('method not implemented yet. Or check the spelling')
                                break

                            regModel.fit(trn.iloc[:,:-1], trn.iloc[:,-1])

                            y_ts_h = regModel.predict(tst.iloc[:,:-1])
                            y_tr_h = regModel.predict(trn.iloc[:,:-1])

                            if NORMALIZE_Y: 
                                y_tr_h = scalerY.inverse_transform(y_tr_h)
                                y_ts_h = scalerY.inverse_transform(y_ts_h)
                                trn.iloc[:,-1] = scalerY.inverse_transform(trn.iloc[:,-1])
                                tst.iloc[:,-1] = scalerY.inverse_transform(tst.iloc[:,-1])

                            mse = np.sqrt(mean_squared_error(tst.iloc[:,-1], y_ts_h))
                            r2 = r2_score(tst.iloc[:,-1], y_ts_h)

                            # print(mse,r2)

                            t_mse = np.sqrt(mean_squared_error(trn.iloc[:,-1], y_tr_h))
                            t_r2 = r2_score(trn.iloc[:,-1], y_tr_h)

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
                            
                            del trn, tst, trn_, tst_, leg_whole_

                    
                    
    save_obj(summ, SAVEFOLDER + MODELNAME)
    return summ



def run_baselines_particle_size(data, **kwargs):
    
    SEP_METHOD = kwargs['SEP_METHOD']
    SEA = kwargs['SEA']
    NUM_REP = kwargs['NUM_REP']
    LEG_P = kwargs['LEG_P']
    VARSET = kwargs['VARSET']
    METHODS = kwargs['METHODS']
    NORMALIZE_Y = kwargs['NORMALIZE_Y']
    SAVEFOLDER = kwargs['SAVEFOLDER']
    MODELNAME = kwargs['MODELNAME']
    SAVE_EACH_RUN = kwargs['SAVE_EACH_RUN']
    
    if not os.path.isdir(SAVEFOLDER):
        os.mkdir(SAVEFOLDER)

    if os.path.exists(SAVEFOLDER + MODELNAME): 
        print("file exists, overwriting")
    
    summ = {}
    
    for sep_method in SEP_METHOD: 
        # print(sep_method)
        for leg in LEG_P: 
            for sea in SEA: 
                for varset in VARSET:
                    for meth in METHODS:
                        nre = 0
                        while nre < NUM_REP:

                            string_exp = sea + '_leg_' + str(leg) + '_' + sep_method + '_' + meth + '_' +  varset + '_' + str(nre)
                            #print(string_exp)
                            nre += 1 

                            # if (varset.lower() == 'full' and leg != 1):
                            #     continue


                            if varset.lower() == 'full':
                                cols_total = ['hs', 'tp', 'steep', 'phase_vel', 'age', 'wind', 'parbin']
                                cols_wind  = ['hs_w', 'tp_w', 'steep_w', 'phase_vel_w', 'age_w', 'wind', 'parbin']
                            elif varset.lower() == 'nowind': 
                                cols_total = ['hs', 'tp', 'steep', 'phase_vel', 'parbin']
                                cols_wind  = ['hs_w', 'tp_w', 'steep_w', 'phase_vel_w', 'parbin']
                            elif varset.lower() == 'reduced':
                                cols_total = ['hs', 'tp', 'wind', 'parbin']
                                cols_wind  = ['hs_w', 'tp_w', 'wind', 'parbin']

                            if sea.lower() == 'total':
                                leg_whole_ = data.loc[data['leg'] == leg, cols_total].dropna().copy()
                            
                            elif sea.lower() == 'wind':
                                leg_whole_ = data.loc[data['leg'] == leg, cols_wind].dropna().copy()
                            
                            s1, s2 = leg_whole_.shape

                            separation=2.0/3    
                            trn_size = ceil(s1*separation) #; print(trn_size, s1)

                            if sep_method.lower() == 'prediction':
                            #     print('training data until ' + str(separation) + ', then test.')
                                trn = leg_whole_.iloc[:trn_size,:].copy()
                                tst = leg_whole_.iloc[trn_size:,:].copy()

                            elif sep_method.lower() == 'interpolation': 
                            #     print('training data random %f pc subset, rest test'%(separation*100))
                                leg_whole_ = shuffle(leg_whole_)
                                trn = leg_whole_.iloc[:trn_size,:].copy()
                                tst = leg_whole_.iloc[trn_size:,:].copy()
                            
                            #print('training / test sizes (2/3 training split): tr %f, ts %f'%(len(trn.iloc[:,1]),len(tst.iloc[:,1])))

                            # Standardize data to 0 mean unit variance based on training statistics (stationarity)
                            # ----------
                            scalerX = preprocessing.StandardScaler().fit(trn.iloc[:,:-1])

                            if NORMALIZE_Y: 
                                scalerY = preprocessing.StandardScaler().fit(trn.iloc[:,-1].values.reshape(-1, 1))

                            trn_ = pd.DataFrame(scalerX.transform(trn.iloc[:,:-1]), columns=trn.iloc[:,:-1].columns, index=trn.index)
                            tst_ = pd.DataFrame(scalerX.transform(tst.iloc[:,:-1]), columns=tst.iloc[:,:-1].columns, index=tst.index)

                            if NORMALIZE_Y: 
                                scalerY = preprocessing.StandardScaler().fit(trn.iloc[:,-1].values.reshape(-1, 1))
                                y_trn = scalerY.transform(trn.iloc[:,-1].values.reshape(-1, 1))
                                y_tst = scalerY.transform(tst.iloc[:,-1].values.reshape(-1, 1))
                            else: 
                                y_trn = trn.iloc[:,-1]
                                y_tst = tst.iloc[:,-1]

                            trn = trn_; 
                            trn['parbin'] = y_trn
                            tst = tst_; 
                            tst['parbin'] = y_tst

                            # y_gt_scaled = pd.DataFrame(scaler.transform(leg_whole_), columns=leg_whole_.columns, index=leg_whole_.index)
                            # y_gt_scaled = y_gt_scaled.iloc[:,-1]
                            # ------------
                            ######### 1 : Ridge Regression
                            if meth.lower() == 'ridgereg':
                                MSE_error = make_scorer(mean_squared_error, greater_is_better=False)
                                regModel = RidgeCV(alphas=np.logspace(-6,6,13), fit_intercept=not NORMALIZE_Y, normalize=False, store_cv_values=False, gcv_mode='svd', cv=5, scoring=MSE_error).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
                            elif meth.lower() == 'pls':
                                n = 3
                                regModel = PLSRegression(n_components=n, scale=False).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
                                regModel.coef_ = np.squeeze(np.transpose(regModel.coef_))
                            elif meth.lower() == 'lasso':
                                regModel = LassoCV(alphas=np.logspace(-3,-1,3), n_alphas=200, fit_intercept=not NORMALIZE_Y, cv=5).fit(trn.iloc[:,:-1], trn.iloc[:,-1])

                            elif meth.lower() == 'lingpr':
                                kernel = DotProduct(sigma_0=1,  sigma_0_bounds=(1e-05, 1e05)) + \
                                     1.0 * WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-3, 1e+3))
                                regModel = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', alpha=0, n_restarts_optimizer=5).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
                                #parameters = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'sigma_0':[0.01, 1, 10]}

                            elif meth.lower() == 'rbfgpr':
                                kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + \
                                     1.0 * WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-3, 1e+3))
                                regModel = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', alpha=0, n_restarts_optimizer=5).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
                            else: 
                                print('method not implemented yet. Or check the spelling')
                                break

                            # regModel.fit(trn.iloc[:,:-1], trn.iloc[:,-1])

                            y_ts_h = regModel.predict(tst.iloc[:,:-1])
                            y_tr_h = regModel.predict(trn.iloc[:,:-1])

                            if NORMALIZE_Y: 
                                y_tr_h = scalerY.inverse_transform(y_tr_h)
                                y_ts_h = scalerY.inverse_transform(y_ts_h)
                                trn.iloc[:,-1] = scalerY.inverse_transform(trn.iloc[:,-1])
                                tst.iloc[:,-1] = scalerY.inverse_transform(tst.iloc[:,-1])

                            mse = np.sqrt(mean_squared_error(tst.iloc[:,-1], y_ts_h))
                            r2 = r2_score(tst.iloc[:,-1], y_ts_h)

                            # print(mse,r2)

                            t_mse = np.sqrt(mean_squared_error(trn.iloc[:,-1], y_tr_h))
                            t_r2 = r2_score(trn.iloc[:,-1], y_tr_h)


                            tr_ys = pd.DataFrame({'gt': trn.iloc[:,-1], 'y_hat': y_tr_h}, index=trn.index)
                            ts_ys = pd.DataFrame({'gt': tst.iloc[:,-1], 'y_hat': y_ts_h}, index=tst.index)
                            
                            if hasattr(regModel, 'alpha_') & hasattr(regModel, 'coef_'):
                                summ[string_exp] = {'regularizer': regModel.alpha_, 
                                                    'weights': regModel.coef_,
                                                    'tr_RMSE': t_mse,
                                                    'tr_R2': t_r2, 
                                                    'ts_RMSE': mse, 
                                                    'ts_R2': r2, 
                                                    'y_tr_hat': tr_ys,
                                                    'y_ts_hat': ts_ys}
                                
                            elif hasattr(regModel, 'coef_') & ~hasattr(regModel, 'alpha_'):
                                summ[string_exp] = {'weights': regModel.coef_, 
                                                    'tr_RMSE': t_mse, 
                                                    'tr_R2': t_r2, 
                                                    'ts_RMSE': mse, 
                                                    'ts_R2': r2, 
                                                    'y_tr_hat': tr_ys,
                                                    'y_ts_hat': ts_ys}
                            else:
                                summ[string_exp] = {'tr_RMSE': t_mse,
                                                    'tr_R2': t_r2, 
                                                    'ts_RMSE': mse, 
                                                    'ts_R2': r2, 
                                                    'y_tr_hat': tr_ys,
                                                    'y_ts_hat': ts_ys}
                                
                            del trn, tst, trn_, tst_, leg_whole_

                    
    if SAVE_EACH_RUN:
        save_obj(summ, SAVEFOLDER + MODELNAME)
    
    return summ