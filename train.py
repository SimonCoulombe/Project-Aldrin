# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:33:49 2021

@author: mleong
"""
import numpy as np
#import pandas as pd
import h2o
from h2o.estimators import H2OGradientBoostingEstimator  
from h2o.grid.grid_search import H2OGridSearch
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from Data_Handler import logging as lg
from Data_Handler.MyPickle import load, dump
from Data_Handler.Import_Data import import_data


import warnings
warnings.filterwarnings('ignore')


def assign_target(df):
    '''
    This is to align the positive and negative class with the evaluation package.
    where Not ranked top (df['target']!=1) is the positive class.
    '''
    df.rename(columns={'partnerrankposition' : 'target'}, inplace=True) 
    
    conditions = [
            (df['target']==1), 
            (df['target']!=1)
            ]
    choices = [1,0]
    df['target'] = np.select(conditions, choices, default=np.nan)
    df['target'] = df['target'].astype('int8')
    return df
    
def h2o_frame(df):
    '''
    Thi step is to ensure some variable is categorical due to Earnix requirement.
    '''
    hf = h2o.H2OFrame(df)
    
    enum_ls = ['target','RATING_AREA_OWN_DAMAGE','YD_LICENCE_TYPE','POSTCODE_STR',
                'METHOD_OF_PARKING','USE_CODE','RATING_AREA_THIRD_PARTY','RATING_AREA_WEATHER',
                'CORPCD','RATARE','MOPCDE','USECDE','RATHIR','RAWEAT','VEHMDL'
                ]
    for each in enum_ls:
        try:
            hf[each] = hf[each].asfactor()
        except:
            pass

    return hf

def training(hf_train, hf_valid, hyperparameter_tunning=False, production=True):
    '''
    1. If tuning==True then the model will tune for hyperparameter, within the predefine space.
    
    2. If tuning==False, the hyperparameter last used for the previous model will be redeployed in 
    the refitting process.
    
    
    Trained models are automatically save in the production folder unless specify.
    
    '''
    if hyperparameter_tunning==True:
        params = {
              'ntrees'                : [i for i in  range(960, 1060, 1)],
              'max_depth'             : [i  for i in range(11, 15, 1)],
              'min_rows'              : [1],     # min_sample_leaf
              'nbins'                 : [i  for i in range(60, 90, 1)],
              'nbins_cats'            : [i  for i in range(650, 800, 1)],  
              'learn_rate'            : [1/i for i in  range(1, 100, 1)],
              'learn_rate_annealing'  : [i for i in  np.arange(0.95, 1, 0.01)],
              'stopping_rounds'       : [i for i in  range(1, 11)],
             }
        
        search_criteria = {'strategy': 'RandomDiscrete', 'max_runtime_secs': 5*60*60}
        grid_search = H2OGridSearch(H2OGradientBoostingEstimator(seed=42), 
                                    grid_id=lg.get_run_id(),
                                    hyper_params=params,
                                    search_criteria=search_criteria,
                                    export_checkpoints_dir = lg.root_logdir/'params')
        
        grid_search.train(y='target',
                  weights_column='weight',
                  training_frame= hf_train,
                  validation_frame= hf_valid,
                  )
        
        grid_search.get_grid(sort_by='logloss', decreasing=False)
        best_model = grid_search.models[0]
        dump(grid_search.get_hyperparams_dict(id=best_model.model_id), 'best_params.pkl')
        lg.dump_mojo(best_model, production=True)
        
        return best_model
    
    else:
        model = H2OGradientBoostingEstimator(seed=42)
        params = load('best_params.pkl')

        model.set_params(**params)
        model.train(y='target',
                weights_column='weight',
                training_frame= hf_train,
                validation_frame= hf_valid,
                model_id='refit'
                )
        lg.dump_mojo(model, production=True)
        return model

def _getgrid(path):
    '''
    path is url to my_logs/params/yyyy-mm-dd for hyperparameter log
    e.g. path = r'my_logs/params/2022-01-24'
    '''
    grid_search = h2o.load_grid(path) # Please locate latest log file
   
    return grid_search.sorted_metric_table()
 

def plt_learning_curve(model):
    '''This is use to plot the training and validation loss agaist epochs '''
    import matplotlib.pyplot as plt
    lc =  model.scoring_history()[['training_logloss','validation_logloss']]
    epochs = len(lc)
    
    x_axis = range(0, epochs)
    
    
    fig, ax = plt.subplots()
    ax.plot(x_axis, lc['training_logloss'], label='Train')
    ax.plot(x_axis, lc['validation_logloss'] , label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.xlabel('epochs')
    plt.title('GBM Log Loss')
    plt.show()
    
def _val_input():
    '''
    This is to validate user input [False/True]
    '''
    while True:
        try:
            hyperparameter_tunning = input('REQUIRE Hyperparameter Tuning? [False/True] : ')
            hyperparameter_tunning = {"true":True,"false":False}[hyperparameter_tunning.lower()]
            return hyperparameter_tunning
        except KeyError:
           print("Invalid input please enter True or False!")
           hyperparameter_tunning=None

    

if __name__ == '__main__':
    
    tuning = _val_input()
    
    #### 1. Initialised H2O instance
    h2o.init(nthreads=-1, max_mem_size='8g')
    
    #### 2. Import Data
    import_cls = import_data('2.Model_refit_Earnixlog.sql')
    
    df = import_cls.df
    df = assign_target(df)
    
    #### 3. Count labels
    df['target'].value_counts(normalize=True)
    
    #### 4. Train test split
    X_train, X_test, y_train, y_test = train_test_split(
                                         df,
                                         df['target'],
                                         test_size=0.3, 
                                         random_state=42,
                                         stratify = df['target']
                                                      )
    X_train['target'].value_counts(normalize=True)
    X_test['target'].value_counts(normalize=True)
    
    X_train['weight'] = class_weight.compute_sample_weight( class_weight='balanced', y=X_train['target'])
    hf_train = h2o_frame(X_train)
    hf_valid = h2o_frame(X_test)
    
    #### 5. Training Model
    training(hf_train, hf_valid, hyperparameter_tunning=True)
    
    h2o.shutdown()
################################################
    #model.get_params()




