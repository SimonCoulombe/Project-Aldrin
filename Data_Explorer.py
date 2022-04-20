# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 13:53:45 2021

@author: mleong
"""

import pandas as pd
from pathlib import Path
root_logdir = Path('my_logs')


class data_explore():
    @classmethod
    def attribute(cls, df):
        all_attribs = list(df)
        #all_attribs.remove('target')
        
        cat_att = []   
        num_att = []
        for i,x in enumerate(all_attribs):   
            if df[x].dtypes == 'O':
                cat_att.append(x)
            elif df[x].dtypes == 'float64' or df[x].dtypes =='int64':
                num_att.append(x)
        return cat_att, num_att    

    @classmethod
    def cor_matrix(cls, df):
        file_path = root_logdir/('corr(%s).xlsx' % df.name) 
        cor_matrix = df.corr()
        with pd.ExcelWriter(file_path ) as writer:   
            cor_matrix.to_excel(writer, sheet_name='correlation')


    def __init__(self, df):    
        self.cat_att, self.num_att = data_explore.attribute(df)
        

    def df_info(self, df):
        file_path = root_logdir/('info(%s).txt' % df.name)
        with open(file_path, 'w') as text_file:        
            df.info(verbose=True, show_counts=True, buf=text_file)
    
    def dtypes_count(self, df):
        file_path = root_logdir/('dtypes(%s).txt' % df.name)
        with open(file_path, 'w') as text_file:
            print(df.dtypes.value_counts(), file=text_file)            

    def stats(self, df):     
        missing = pd.DataFrame()
        missing['Percentage %'] = df.isnull().sum()/len(df.index)
        missing['Count'] = df.isnull().sum()
        
        file_path = root_logdir/('stats(%s).xlsx' % df.name)
        with pd.ExcelWriter(file_path) as writer:   
            df[self.num_att].describe().to_excel(writer, sheet_name='describe')
            missing.to_excel(writer, sheet_name='missing count')

    def cardinality(self, df):
        file_path = root_logdir/('cardinality(%s).txt' % df.name)
        with open(file_path, 'w') as text_file:
            for x in self.cat_att:
                cat = df[x].value_counts()/len(df.index)*100
                print ("="*20,' Percentage % : ',x, file=text_file)
                print (cat.to_string(), file=text_file)

    