import pandas as pd
import numpy as np
import pickle
import os

from datetime import datetime 
from math import ceil

import ace_package.dataset as dataset 
import ace_package.modeling as modeling
from ace_package.modeling import plot_predicted_timeseries

import GPy
import sklearn as sk
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kern# DotProduct, RBF, WhiteKernel
from sklearn.model_selection import GridSearchCV

def save_obj(obj, fname):
    with open(fname + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def run_baselines_particle_size(data, **kwargs):
    
    SEP_METHOD = kwargs['SEP_METHOD']
    SEA = kwargs['SEA']
    NUM_REP = kwargs['NUM_REP']
    LEG_P = kwargs['LEG_P']
    VARSET = kwargs['VARSET']
    METHODS = kwargs['METHODS']
    NORMALIZE_Y = kwargs['NORMALIZE_Y']
    NORMALIZE_X = kwargs['NORMALIZE_X']
    SAVEFOLDER = kwargs['SAVEFOLDER']
    MODELNAME = kwargs['MODELNAME']
    SPLIT_SIZE = kwargs['SPLIT_SIZE']
    
    if not os.path.isdir(SAVEFOLDER):
        os.mkdir(SAVEFOLDER)

    if os.path.exists(SAVEFOLDER + MODELNAME): 
        print("file exists, overwriting")
    
    summ = {}
    
    for sep_method in SEP_METHOD: 
        #print(sep_method)
        for leg in LEG_P: 
            #print(leg)
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

                            separation=SPLIT_SIZE
                            trn_size = ceil(s1*separation) #; 
                            # print(trn_size, s1)

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
                            inds_trn = trn.index

                            # SCALE TRAINING DATA X, y
                            if NORMALIZE_X:
                                scalerX = preprocessing.StandardScaler().fit(trn.iloc[:,:-1])
                                trn = pd.DataFrame(scalerX.transform(trn.iloc[:,:-1]), columns=trn.iloc[:,:-1].columns, index=trn.index)
                            else:
                                trn = trn.iloc[:,:-1]
                                
                            if NORMALIZE_Y: 
                                scalerY = preprocessing.StandardScaler().fit(trn.iloc[:,-1].values.reshape(-1, 1))
                                trn['parbin'] = y_trn = scalerY.transform(trn.iloc[:,-1].values.reshape(-1, 1))
                            else: 
                                trn['parbin'] = y_trn = trn.iloc[:,-1]

                            # ------------
                            ######### 1 : Ridge Regression
                            if meth.lower() == 'ridgereg':
                                MSE_error = make_scorer(mean_squared_error, greater_is_better=False)
                                regModel = RidgeCV(alphas=np.logspace(-6,6,13), fit_intercept=not NORMALIZE_Y, 
                                                   normalize=False, store_cv_values=False, gcv_mode='svd',
                                                   cv=5, scoring=MSE_error).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
                            elif meth.lower() == 'pls':
                                n = 3
                                regModel = PLSRegression(n_components=n, scale=False).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
                                regModel.coef_ = np.squeeze(np.transpose(regModel.coef_))
                            elif meth.lower() == 'lasso':
                                regModel = LassoCV(alphas=np.logspace(-3,-1,3), n_alphas=200, 
                                                   fit_intercept=not NORMALIZE_Y, cv=5).fit(trn.iloc[:,:-1], trn.iloc[:,-1])

                            elif meth.lower() == 'lingpr':
                                kernel = DotProduct(sigma_0=1,  sigma_0_bounds=(1e-05, 1e05)) + \
                                     1.0 * WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-3, 1e+3))
                                regModel = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', 
                                                                    alpha=0, n_restarts_optimizer=5).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
                            elif meth.lower() == 'rf':
                                import sklearn.ensemble
                                regModel = sklearn.ensemble.RandomForestRegressor(n_estimators=100, criterion='mse', 
                                            max_depth=10, min_samples_split=2, min_samples_leaf=1).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
                
                            elif meth.lower() == 'rbfgpr':
                                kernel = 1.0 * kern.RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + \
                                1.0 * kern.WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-1, 1e+4)) + \
                                1.0 * kern.ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-05, 100000.0)) + \
                                1.0 * kern.DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-05, 100000.0))
                                    
                                regModel = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', 
                                                                    alpha=0, 
                                                                    n_restarts_optimizer=5).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
                            elif meth.lower() == 'rbfgprard':
                                
                                #x = trn.iloc[:,:-1].values
                                #y = trn.iloc[:,-1].values.reshape(-1,1)
                                s1 = trn.iloc[:,:-1].values.shape[1]
                                
                                k = (GPy.kern.RBF(s1, ARD=True)
                                     + GPy.kern.White(s1, 0.001) 
                                     + GPy.kern.Bias(s1, 0.001))
                                     #+ GPy.kern.Linear(s1, variances=0.001, ARD=False))

                                regModel = GPy.models.GPRegression(trn.iloc[:,:-1].values, 
                                                                   trn.iloc[:,-1].values.reshape(-1,1), 
                                                                   kernel=k)
                                #regModel.optimize_restarts(parallel=True, robust=True, num_restarts=5, max_iters=200)
                                regModel.optimize('scg', max_iters=200) # 'scg'
                #                 print(regModel)
                                regModel.coef_ = regModel.sum.rbf.lengthscale
                    
                            else: 
                                print('method not implemented yet. Or check the spelling')
                                break

                            inds_tst = tst.index
                            
                            if NORMALIZE_X:
                                tst_ = pd.DataFrame(scalerX.transform(tst.iloc[:,:-1]), columns=tst.iloc[:,:-1].columns, index=inds_tst)
                            else:
                                tst_ = tst.iloc[:,:-1]

                            if meth.lower() == 'rbfgprard': 
                                # x_ = tst_.values
                                # x  = trn.iloc[:,:-1].values
                                
                                y_ts_h = regModel.predict(tst_.values)[0].reshape(-1,)
                                print(y_ts_h.shape)
                                y_ts_h = pd.Series(y_ts_h,index=inds_tst)
                                y_tr_h = regModel.predict(trn.iloc[:,:-1].values)[0].reshape(-1,)
                                print(y_tr_h.shape)
                                y_tr_h = pd.Series(y_tr_h,index=inds_trn)
                            else: 
                                y_ts_h = regModel.predict(tst_).reshape(-1,)
                                print(y_ts_h.shape)
                                y_ts_h = pd.Series(y_ts_h,index=inds_tst)
                                y_tr_h = regModel.predict(trn.iloc[:,:-1]).reshape(-1,)
                                print(y_tr_h.shape)
                                y_tr_h = pd.Series(y_tr_h,index=inds_trn)

                          
                            if NORMALIZE_Y: 
                                y_tr_h = scalerY.inverse_transform(y_tr_h)
                                y_ts_h = scalerY.inverse_transform(y_ts_h)
                                y_tr_gt = scalerY.inverse_transform(trn.iloc[:,-1])
                                y_ts_gt = scalerY.inverse_transform(tst.iloc[:,-1])
                            #else: 
                            #    y_tr_gt = trn.iloc[:,-1]
                            #    y_ts_gt = tst.iloc[:,-1]
                
                            # Compute scores 
                            mse = np.sqrt(mean_squared_error(tst.iloc[:,-1], y_ts_h))
                            r2 = r2_score(tst.iloc[:,-1], y_ts_h)

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
                            
                            del trn, tst, tst_, leg_whole_, regModel

                    
                    
    save_obj(summ, SAVEFOLDER + MODELNAME)
    return summ