# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:01:22 2021

@author: mleong
"""
from bs4 import BeautifulSoup as bs
import requests
import json
import pandas as pd
from pathlib import Path
from Data_Handler.Import_Data import import_data

def parse_json():
    path = Path('scoring/'+ 'Earnix New - Validated.postman_collection.json')
    with open(path) as json_file:
        json_file = json.load(json_file)
    
    
    dictionary = {}
    
    for item in json_file['item']:
        request = item['request']
        headers_dict = request['header'][0] 
        
        name = item['name']
        body = request['body']['raw']
        url = request['url']['raw']
        headers = {headers_dict['key'] : headers_dict['value']}
        
        dictionary[name] = {'url'     : url,
                            'headers' : headers,
                            'body'    : body
                            }
    return dictionary

def post_request(d):
    response = requests.post(d['url'], data=d['body'], headers=d['headers'])
    return response.content

def import_csv():
    '''
    Import as string only
    '''
    df = pd.read_csv("T-8.csv",dtype=object)
    df = df.rename(columns={'Origination Demand ADJ':'Data_Probability'})
    return df

    
if __name__=="__main__":
    
# =============================================================================
#     #score throgh model
# =============================================================================
    from train import assign_target, h2o_frame
    from scoring.predict import retrieve_models
    import h2o

    h2o.init(nthreads=-1)
    # reimport
    import_cls = import_data('_sample_API.sql')
    df = import_cls.df
    df['partnerrankposition'] = None

    df = assign_target(df)
    hf = h2o_frame(df)
    
    model_history = retrieve_models()
    model = model_history['2. April_v1'] # select model you wanted to test for
    
    results = model.predict(hf).as_data_frame()
    df['p1'] = results['p1']
    
# =============================================================================
#     #score throgh API
# =============================================================================
    #df= import_bq()
    df = df.astype("string")
    
    dictionary = parse_json()
    # extract xml body from dictionary
    bs_body = bs(dictionary['Oceania - S1']['body'], "xml")
    
    tags = bs_body.attributeNames.find_all("string")
    values = bs_body.values.find_all("string")
    tags_values = dict(zip(tags, values))
    
    import time
    t1 = time.time()
    values_ls = []
    # outer loop iterate over each rows in test data
    for df_i, df_row in df.iterrows():
        # inner loop iterate over each required tag and replace value from test data     
        for i, (t,v) in enumerate(tags_values.items()) :
            # try to repalce value if present, otherwise continue looping
            try:
                new_val = df_row[t.contents[0]]
                #print( t.contents[0] ,":", new_val)
                #print(bs_body.values.find_all("string")[i])
                bs_body.values.find_all("string")[i].string.replaceWith(new_val)
       
            except:
                continue
   
        #replace dictionary with new SOAP
        dictionary['Oceania - S1']['body'] = str(bs_body)       
        
        #post to Earnix
        response = post_request(dictionary['Oceania - S1'])
        bs_response = bs(response, "lxml") #turn to beautiful soup object
  
        #parsing output    
        value_list = bs_response.find("values").find_all("string")
        
        values = [i.text for i in value_list]
        values.insert(0, df_row['CLIENT_NUMBER'])
        values.insert(8, str(bs_body))
        
        values_ls.append(values)
        
    print(time.time()-t1)    
        
        
    #processed the appended list
    att_list = bs_response.find("attributenames").find_all("string")
    columns = [i.text for i in att_list]
    columns.insert(0,'CLIENT_NUMBER')
    columns.insert(8,'post_body')
   
    df_resp =  pd.DataFrame(values_ls,  columns =columns)
    # add extra info to output files
    #df_resp.to_csv('earnix_response.csv')

    
# =============================================================================
#     Join
# =============================================================================
    out = df.merge(df_resp, left_on='CLIENT_NUMBER', right_on='CLIENT_NUMBER')
    out.to_csv('earnix_response.csv')
    
