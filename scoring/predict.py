# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 13:59:40 2021

@author: mleong
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import h2o

from Data_Handler.Import_Data import import_data
from Evaluation import model_eval

import warnings
warnings.filterwarnings('ignore')


from train import assign_target, h2o_frame

def retrieve_models():
    '''
    This function download all production model in the directory into a dictionary.
    '''
    directory = Path('scoring/MOJO_PROD')
    
    model_history = {}
    for file in os.listdir(directory):
        model =  h2o.import_mojo( 
                    (directory/file).absolute().as_uri()
                                )
        model.model_id = (directory/file).stem
    
        model_history.update({model.model_id : model})
    return model_history 


if __name__ == '__main__':   
    h2o.init(nthreads=-1)
    
    #### P2. Import Data
    import_cls = import_data('3.Scoring.sql')
    df = import_cls.df
    df['QUOTE_DATE'] = pd.to_datetime(df['QUOTE_DATE'])

    df = assign_target(df)
    hf = h2o_frame(df)
    
    model_history = retrieve_models()   
        
    #### predict
    metric = model_eval(model_history, hf, 40)
    metric.models_stats()
    
    columns = []
    models = list(model_history.keys())
    for i in models:
        column = 'pred_%s' % i
        df[column] = metric.y_pred[i]
        columns.append(column)
    
    ### calculate False Positive Rate (where actual is 1 but predicted to be not)
    # 1.Where actual is rank #1
    rank_1 = df.loc[df['target']==1]
    
    # 2.Columns to aggregate into daily time series
    ts = rank_1.set_index('QUOTE_DATE').resample('D')[columns + ['target']].sum()
    
    # 3. FPR = FP/[FP+TN] = 1- TN/All negative
    for i in models:
        column = 'pred_%s' % i
        ts['FPR_%s' % i] = 1 - ts[column]/ts['target']
    
    from google.cloud import bigquery
    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(
                        write_disposition="WRITE_TRUNCATE",
                    )
    
    load_job = client.load_table_from_dataframe(
                    ts,
                    "pricing-nonprod-687b.ML.AL_FPR",
                    location='australia-southeast1',
                    job_config = job_config,
                    )
    load_job.result()
      


# =============================================================================
#from sklearn.metrics import confusion_matrix
#b = confusion_matrix(df['predicted@2021-12-31'],df['predicted@2021-12-31']) 
# 
# df = pd.read_csv('scoring/RATARE_MarketingFund.csv')
# df['MKT_FUND'] = df['Relativity']-1
# a = df.set_index('RATARE')['MKT_FUND'].to_dict()
# =============================================================================
