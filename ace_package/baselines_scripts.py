import pandas as pd
import numpy as np
import pickle
import os

from datetime import datetime 
from math import ceil

import ace_package.dataset as dataset 
import ace_package.modeling as modeling
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

def run_baselines_particle_size(data, **kwargs):
    '''
    ADD EXHAUSTIVE EXPLANATION OF OPTIONS HERE
    
    '''
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
    N_TST = kwargs['N_TST']
    SPLIT_SIZE = kwargs['SPLIT_SIZE']
    SPLIT_MODE = kwargs['SPLIT_MODE']
    #TRNTST_SPLIT = kwargs['TRNTST_SPLIT'] # either [] -- defaults to 2/3 training, [1./2] -- specify fraction 
                                          # or [...] provides indexed array for training (1) and testing points (2, 3, 4, ...)
    
    #if len(TRNTST_SPLIT) == 1:
    #    trntst_split = TRNTST_SPLIT#; print(trn_size, s1)
    #    INDEX_TRIALS = 1
    #elif len(TRNTST_SPLIT) == 0: 
    #    trntst_split = 2.0/3
    #    INDEX_TRIALS = 1
    #elif len(TRNTST_SPLIT) > 1:
    #    trntst_split = [0]
    #    SEP_METHOD = 'index' # 1 for training, 2 for testing
#        if len(TRNTST_SPLIT.shape) == 1:
#            INDEX_TRIALS = 1
#        elif len(TRNTST_SPLIT.shape) == 2:
     #   INDEX_TRIALS = TRNTST_SPLIT.shape[1]
     #   print(INDEX_TRIALS)

    if not os.path.isdir(SAVEFOLDER):
        os.mkdir(SAVEFOLDER)

    if os.path.exists(SAVEFOLDER + MODELNAME): 
        print("file exists, overwriting")
    
    summ = {}
    
    for sep_method in SEP_METHOD: 
         for leg in LEG_P: 
            for sea in SEA: 
                for varset in VARSET:
                    for meth in METHODS:
                        nre = 0
                        while nre < NUM_REP:

                            string_exp_main = sea + '_leg_' + str(leg) + '_' + sep_method + '_' + meth + '_' +  varset

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
                                sea_subset = data.loc[:, cols_total].dropna().copy()

                            elif sea.lower() == 'wind':
                                sea_subset = data.loc[:, cols_wind].dropna().copy()    
    
                            if 0:
                                scalerX = preprocessing.StandardScaler().fit(sea_subset.iloc[:,:-1])

                                if NORMALIZE_Y:
                                    scalerY = preprocessing.StandardScaler().fit(sea_subset.iloc[:,-1].values.reshape(-1, 1))

                                sea_subset_ = pd.DataFrame(scalerX.transform(sea_subset.iloc[:,:-1]), columns=sea_subset.iloc[:,:-1].columns, index=sea_subset.index)
                                # sea_subset_ = sea_subset
                                
                                if NORMALIZE_Y:
                                    sea_subset_['parbin'] = scalerY.transform(sea_subset.iloc[:,-1].values.reshape(-1, 1))
                                else:
                                    sea_subset_['parbin'] = sea_subset['parbin']

                            
                            if sep_method == 'index':
                                sea_subset_ = sea_subset
                                curr_inds = modeling.sample_trn_test_index(sea_subset_.index,split=SPLIT_SIZE, N_tst=N_TST, mode=SPLIT_MODE)
                                # print(curr_inds)
                                
                            elif sep_method != 'index':
                                sea_subset_ = sea_subset[data.loc[sea_subset.index,:].leg == leg].copy()
                                curr_inds = modeling.sample_trn_test_index(sea_subset_.index,split=SPLIT_SIZE, N_tst='all', mode='final')
                                
                            print(curr_inds)
                            #print(np.unique(curr_inds.ind,return_counts=True))
                            #print(sea_subset_.head())
                            #print(sea_subset_.isnull().sum())
                            # print(curr_inds.head(-5))

                            for it_num, tst_subset in enumerate(np.setdiff1d(np.unique(curr_inds),1)): # 1 is reserved for training
                                tst_subset = int(tst_subset)
                                
                                # print(it_num,tst_subset)
                                #print((curr_inds == 1).sum())
                                
                                
                                if sep_method != 'index':
                                    # print(sep_method,sep_method != 'index')

                                    string_exp = string_exp_main + '_' + str(nre)
                                    leg_whole_ = sea_subset_#.loc[data['leg'] == leg,:]

                                    # s1 = leg_whole_.shape[0]
                                    # trn_size = ceil(s1*SPLIT_SIZE) #; print(trn_size, s1)

                                    if sep_method.lower() == 'prediction':
                                    #     print('training data until ' + str(separation) + ', then test.')
                                        #tr = leg_whole_.iloc[:trn_size,:].copy()
                                        #ts = leg_whole_.iloc[trn_size:,:].copy()
                                        tr = leg_whole_.loc[curr_inds.ind == 1,:].copy()
                                        ts = leg_whole_.loc[curr_inds.ind == tst_subset,:].copy()

                                    elif sep_method.lower() == 'interpolation': 
                                    #     print('training data random %f pc subset, rest test'%(separation*100))
                                        leg_whole_ = shuffle(leg_whole_)
                                        tr = leg_whole_.iloc[:int(np.floor(SPLIT_SIZE*len(leg_whole_))),:].copy()
                                        ts = leg_whole_.iloc[int(np.floor(SPLIT_SIZE*len(leg_whole_))):,:].copy()
                                        #tr = leg_whole_.loc[curr_inds.ind == 1,:].copy()
                                        #ts = leg_whole_.loc[curr_inds.ind == tst_subset,:].copy()
                                        
                                elif sep_method == 'index': 

                                    string_exp = string_exp_main + '_id_tst_split_' + str(tst_subset-1) + '_' + str(nre)

                                    tr = sea_subset_.iloc[np.where(curr_inds.ind == 1)[0],:].copy()
                                    ts = sea_subset_.iloc[np.where(curr_inds.ind == tst_subset)[0],:].copy()

                                else: 
                                    error(sep_method + ' not specified')


                                #print(it_num, tst_subset, string_exp)
                                #print(trn.shape[0],tst.shape[0])

                                #print('training / test sizes (2/3 training split): tr %f, ts %f'%(len(trn.iloc[:,1]),len(tst.iloc[:,1])))

                                # Standardize data to 0 mean unit variance based on training statistics (stationarity)
                                # ----------
                                if 1:
                                    if it_num == 0:
                                        scalerX = preprocessing.StandardScaler().fit(tr.iloc[:,:-1])

                                        if NORMALIZE_Y: 
                                            scalerY = preprocessing.StandardScaler().fit(tr.iloc[:,-1].values.reshape(-1, 1))

                                    trn_ = pd.DataFrame(scalerX.transform(tr.iloc[:,:-1]), columns=tr.iloc[:,:-1].columns, index=tr.index)
                                    tst_ = pd.DataFrame(scalerX.transform(ts.iloc[:,:-1]), columns=ts.iloc[:,:-1].columns, index=ts.index)

                                if NORMALIZE_Y: 
                                    # scalerY = preprocessing.StandardScaler().fit(trn.iloc[:,-1].values.reshape(-1, 1))
                                    y_trn = scalerY.transform(tr.iloc[:,-1].values.reshape(-1, 1))
                                    y_tst = scalerY.transform(ts.iloc[:,-1].values.reshape(-1, 1))
                                else: 
                                    y_trn = tr.iloc[:,-1]
                                    y_tst = ts.iloc[:,-1]


                                trn = tr    
                                trn['parbin'] = y_trn
                                tst = ts; 
                                tst['parbin'] = y_tst

                                # y_gt_scaled = pd.DataFrame(scaler.transform(leg_whole_), columns=leg_whole_.columns, index=leg_whole_.index)
                                # y_gt_scaled = y_gt_scaled.iloc[:,-1]
                                # ------------
                                if it_num == 0:
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
                                    elif meth.lower() == 'rbfgpr':
                                        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + \
                                             1.0 * WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-1, 1e+4))
                                        regModel = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', 
                                                                            alpha=0, n_restarts_optimizer=5).fit(trn.iloc[:,:-1], trn.iloc[:,-1])
                                    else: 
                                        print('method not implemented yet. Or check the spelling')
                                        break

                                # regModel.fit(trn.iloc[:,:-1], trn.iloc[:,-1])

                                y_ts_h = regModel.predict(tst.iloc[:,:-1])
                                y_tr_h = regModel.predict(trn.iloc[:,:-1])

                                if NORMALIZE_Y: 
                                    y_tr_h = scalerY.inverse_transform(y_tr_h)
                                    y_ts_h = scalerY.inverse_transform(y_ts_h)
                                    y_tr_gt = scalerY.inverse_transform(trn.iloc[:,-1])
                                    y_ts_gt = scalerY.inverse_transform(tst.iloc[:,-1])
                                else: 
                                    y_tr_gt = trn.iloc[:,-1]
                                    y_ts_gt = tst.iloc[:,-1]
                                                                        
                                                                        
                                mse = np.sqrt(mean_squared_error(y_ts_gt, y_ts_h))
                                r2 = r2_score(y_ts_gt, y_ts_h)

                                # print(mse,r2)

                                t_mse = np.sqrt(mean_squared_error(y_tr_gt, y_tr_h))
                                t_r2 = r2_score(y_tr_gt, y_tr_h)


                                # tr_ys = pd.DataFrame({'gt': y_tr_gt, 'y_hat': y_tr_h}, index=trn.index)
                                # ts_ys = pd.DataFrame({'gt': y_ts_gt, 'y_hat': y_ts_h}, index=tst.index)

                                if hasattr(regModel, 'alpha_') & hasattr(regModel, 'coef_'):
                                    summ[string_exp] = {'regularizer': regModel.alpha_, 
                                                        'weights': regModel.coef_,
                                                        'tr_RMSE': t_mse,
                                                        'tr_R2': t_r2, 
                                                        'ts_RMSE': mse, 
                                                        'ts_R2': r2}#, 
                                                        #'y_tr_hat': tr_ys,
                                                        #'y_ts_hat': ts_ys}

                                elif hasattr(regModel, 'coef_') & ~hasattr(regModel, 'alpha_'):
                                    summ[string_exp] = {'weights': regModel.coef_, 
                                                        'tr_RMSE': t_mse, 
                                                        'tr_R2': t_r2, 
                                                        'ts_RMSE': mse, 
                                                        'ts_R2': r2}#, 
                                                        #'y_tr_hat': tr_ys,
                                                        #'y_ts_hat': ts_ys}
                                else:
                                    summ[string_exp] = {'tr_RMSE': t_mse,
                                                        'tr_R2': t_r2, 
                                                        'ts_RMSE': mse, 
                                                        'ts_R2': r2}#, 
                                                        #'y_tr_hat': tr_ys,
                                                        #'y_ts_hat': ts_ys}

                                # del trn, tst, trn_, tst_, leg_whole_
                            
                            nre += 1

                    
    if SAVE_EACH_RUN:
        save_obj(summ, SAVEFOLDER + MODELNAME)
    
    return summ