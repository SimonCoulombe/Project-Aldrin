# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:53:45 2021
@author: mleong

Project : Aldrin
"""

import os
import time
from pathlib import Path
sql_dir = Path('Data_Handler/SQL')

from google.cloud import bigquery
os.environ['GOOGLE_APPLICATION_CREDENTIALS'
                   ] = r'C:\Users\mleong\AppData\Roaming\gcloud\application_default_credentials.json'


class import_data():
    @classmethod
    def read_script(cls, file_name):
        file = sql_dir/file_name
        query = file.read_text()
        return query


    def __init__(self, file_name):
        self.sql =  import_data.read_script(file_name)
        self.root_logdir = Path('my_logs')
        
        client = bigquery.Client()
        self.df = client.query(self.sql).to_dataframe()
        self.df.name = file_name
        
    def clean(self):
        print('empty...')
        
    
    def df_info(self):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S") # "run_%Y_%m_%d-%H_%M_%S" for 2 digits second
        
        file_path = self.root_logdir/('info(%s).txt' % self.df.name)
        with open(file_path, 'w+') as file:
            file.write("############## >> FILE IMPORTED ON %s << ################" %
                       timestamp + '\n')
            self.df.info(verbose=True,show_counts=True, buf=file)
    
            file.close()
    

if __name__=='__main__':
    '''
    example of usage
    '''
    data_cls = import_data()
    df = data_cls.df
    data_cls.df_info()
