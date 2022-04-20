# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 15:15:16 2021

@author: mleong

This evaluation package is design to compare multiple classification model concurrently.
For a multilabel problem the label is arrange in the assending order (avoid using label=0), 
so the confusion is as follows:
    
  Class      1     2    3    
    1    [[11502  29     12]
    2     [ 2    5092    23]
    3     [ 8      4   2826]]

and when combine become:
  Class        1   !=1    
    1     [[11502   41]
   !=1     [10    7945]]
   
For a Binary only tranning (0/1), the matrix are flip intentionaly so that the positive class is on the lower right:
 Class        1     0  
    1     [[11502   41]
   0      [10    7945]]

"""

from sklearn.preprocessing import  LabelBinarizer
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
import numpy as np
#from Data_Handler.MyPickle import  dump

import os
root_logdir = os.path.join(os.curdir, 'my_logs')

import warnings
warnings.filterwarnings('ignore')

class model_eval():

    @classmethod
    def plot_precision_recall_vs_threshold(cls, precisions, recalls, thresholds, _id=None):
        plt.plot(thresholds, precisions[:-1], "--", label="Precision-%s" % _id)
        plt.plot(thresholds, recalls[:-1], "-", label="Recall-%s" % _id)
        plt.xlabel("Threshold")
        #plt.legend(loc="lower left")
        plt.ylim([0, 1.1])
        plt.title('Precision/Recall Not Rank #1', y=-0.24, fontweight="bold")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)
    
    @classmethod
    def plot_roc_curve(cls, fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([-0.01, 1, 0, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Not Rank #1', fontweight="bold")
        plt.legend()

    def __init__(self, models, data, BottomPercentile=None, margin_criteria=False):
        '''
        1. models take a dictionary with key value pair, 
                                                    e.g {'model_id1' : model_1
                                                         'model_id2' : model_2}
        2. data take a h2oFrame which contained both attribute and target.
           ** Hence please rename the target column as 'target'.
        3. Define the N bottom percentile of class-1 as not class-1 (TP)   
        '''

        self.y_true =  data['target'].as_data_frame()['target']
        self.models = models
        self.data = data
        
        self.lb = LabelBinarizer()
        self.y_onehot = self.lb.fit_transform(self.y_true)
        self.y = 1-self.y_onehot[:, 0] # not class-1
        
        self._pcntile = BottomPercentile
        self._margin_criteria = margin_criteria
        
        self.y_pred = {}
        self.y_proba = {} # only store the probability positive class (not rank-1)
        self.y_proba_1 = {}
        self.predict()
        
    def predict(self, ):
        '''
        This is the section that need to differ if using native SKlearn framework.
        Due to H2O model take a H2O frame as input and output which needed to be converted to appropriate format
        before sklearn scoring can apply.
        
        ** Please keep the remaining function in this package the same except this function.
        '''
        for i, (_id, model) in enumerate(self.models.items()):
            results = model.predict(self.data).as_data_frame()
            
            # threshold_cut
            threshold = np.percentile(results['p1'], self._pcntile)
            print(threshold)
            results['predict'] = np.where(results['p1']<=threshold, 0, 1)
            
            # only consider for allowable margin
            if self._margin_criteria==True:
                results['MKT_FUND'] = self.data['Marketing_Fund'].as_data_frame()['Marketing_Fund']
                results['predict'] = 1 #mask all result to class-1 before apply condition below
                results['predict'] = np.where( (results['MKT_FUND'] > -0.03)&
                                                (results['p1']<=threshold), 0, 1)
                
            self.y_pred[_id] = results['predict']
            self.y_proba[_id] = 1-results['p1']  # probability not class-1
            self.y_proba_1[_id] = results['p1'] # probability of class-1
            
    
    def precision_recall_plot(self):
        precisions = {} 
        recalls = {}
        thresholds = {}
        for i, (_id, model) in enumerate(self.models.items()):
            precisions[_id], recalls[_id], thresholds[_id] = precision_recall_curve(self.y, self.y_proba[_id])
            model_eval.plot_precision_recall_vs_threshold(precisions[_id],
                                                          recalls[_id],
                                                          thresholds[_id],
                                                          _id)
        plt.show()
        return precisions, recalls, thresholds
        
    
    def compare_matrix_multi(self):        
        for i, (_id, model) in enumerate(self.models.items()):    
            print('''>>>>>>>>>>>>> MULTICLASS MODEL:%s Confusion Matrix <<<<<<<<<''' % _id)
            print(confusion_matrix(self.y_true,  self.y_pred[_id]))
            print('-----------------------------------------')
            print("Weighted Multiclass ROC_AUC   : {:.2%}".format(
                  roc_auc_score(self.lb.transform(self.y_true),  self.lb.transform(self.y_pred[_id]),
                                average='weighted',  multi_class='ovr')))
            print("Specific Class Accuracy       : {:.2%}".format(
                                accuracy_score(self.y_true,  self.y_pred[_id])))
            
            print("\n")
            print("Precision and Recall of each individual class")
            recalls = recall_score(self.y_true,  self.y_pred[_id], average=None)
            precisions = precision_score(self.y_true,  self.y_pred[_id], average=None)
            print("recall    : ", ["{:.2%}".format(x) for x in recalls])  
            print("precision : ", ["{:.2%}".format(x) for x in precisions])
            print('============== END ===================== \n')
            
    def compare_matrix_binary(self):
        for i, (_id, model) in enumerate(self.models.items()):
            CM = confusion_matrix(self.y, self.y_pred[_id] !=1)
            TN = CM[0][0]
            FN = CM[1][0]
            TP = CM[1][1]
            FP = CM[0][1]
            if self._margin_criteria==True:
                print('''******* After Consider for Margin equal to or more than -3% *******''')
            
            print('''>>>>>>>>>> BINARY MODEL:%s with bottom %sth percentile <<<<<<<<<''' % (_id, self._pcntile))
            print('''>>>>>>>>>> Confusion Matrix <<<<<<<<<''')
            print(confusion_matrix(self.y, self.y_pred[_id] !=1))
            
            print('''>>>>>>>>>> Metric @ %s  <<<<<<<<<''' % _id)
            print("ROC_AUC   : {:.2%}".format(roc_auc_score(  self.y, self.y_proba[_id]   )))
            print("False +ve : {:.2%}".format(FP/(FP+TN)))
            print("Precision : {:.2%}".format(TP/(FP+TP)))
            print("Accuracy  : {:.2%}".format( (TP+TN)/(TP+TN+FP+FN) ))
            print("Miss Rate : {:.2%}".format( FN/(TP+FN)))
            print("\n")
    
    def compare_roc_curve(self):
        fpr = {}
        tpr = {}
        for i, (_id, model) in enumerate(self.models.items()):                
            fpr[_id], tpr[_id], thresholds = roc_curve(self.y, self.y_proba[_id])
            model_eval.plot_roc_curve(fpr[_id], tpr[_id], label=_id)    
        plt.show()
        return fpr, tpr
   
    def models_stats(self):
        fpr, tpr = self.compare_roc_curve()
        if self.y_onehot.shape[1] > 1: 
            self.compare_matrix_multi()
        precisions, recalls, thresholds = self.precision_recall_plot()
        self.compare_matrix_binary()
        
        # trigger allowable margin criteria
        self._margin_criteria=True
        self.predict()
        self.compare_matrix_binary()
        
        
# =============================================================================
#         plt_data = {}
#         for i, (_id, model) in enumerate(self.models.items()):
#             plt_data[_id] = {'fpr'       : fpr[_id],
#                              'tpr'       : tpr[_id],
#                              'precision' : precisions[_id],
#                              'recalls'   : recalls[_id],
#                              'thresholds': thresholds[_id]
#                              }
#         dump(plt_data, root_logdir + '\\plt_data.pkl')
# =============================================================================
