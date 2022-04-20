# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 14:42:01 2021

@author: mleong
"""
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
root_logdir = Path('my_logs')
#root_logdir = os.path.join(os.curdir, 'my_logs')
import h2o
    

def get_run_id(timestamp=False):
    if timestamp == True:
        run_id = time.strftime("%Y-%m-%d %H.%M.%S") # "run_%Y_%m_%d-%H_%M_%S" for 2 digits second
    else:
        run_id = time.strftime("%Y_%m_%d") # no timestamp on export file
    return run_id

def save_fig(fig_id, *, tight_layout=True,fig_extension="png", resolution=300):
    path = root_logdir/('fig_id.%s' % fig_extension)
    
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path.absolute().as_uri(), format=fig_extension, dpi=resolution)
    
def dump_mojo(model, *, production=False, timestamp=False):
    run_id = get_run_id(timestamp)
    
    if production == False:
        path = Path('MOJO_DEV/'+ model.model_id +'__'+ run_id + '.zip')
    else:
        path = Path('scoring/MOJO_PROD/' + run_id + '.zip')
      
 
    h2o.api("GET /99/Models.mojo/%s" % model.model_id, data={"dir": path.absolute().as_uri(), "force": False})["dir"]
    print("model saved in ", path)        
    