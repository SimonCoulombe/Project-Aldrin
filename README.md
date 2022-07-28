# Project Aldrin
<a name="TBC"></a>
### Table of Contents

Part I. [Overview](#section-1)  
1. [Data_Handler](Data_Handler/)  
2. [scoring](scoring/)
3. [my_logs](my_logs/)

Part II. [Modules](#section-2)  
1. [Evaluation.py](#evaluation)
2. [Data_Explorer.py](#data-explorer)

Part III. [Notebooks](#section-3)  
1. [train.ipynb ](#train)
2. [R&D.ipynb](#RnD)	


<a name="section-1"></a>
## Overview
Project Aldrin aims to improve the competitiveness of our pricing offered on aggregator (CTM). The following repository contained the predictive model capable of estimating the likelihood of a given quote being ranked number-1 on the aggregator website and the resource needed to reproduce.

In [Part III](#section-3) consist of a jupyter-notebook - [train.ipynb](#train), which is the primary documentation of the procedure. The python environment requirement is listed in [requirements.txt](). Use `pip install -r requirements.txt` to replicate it.

The [Data_Handler](Data_Handler/) is a folder that contains all data handling modules, including SQL scripts, downloading function from Bigquery, data I/O to and from GCP storage bucket, logging capability, i.e. loading or dumping pickle file, and import or export MOJO file.

Data that needed to be logged during the traing process are kept in the [my_logs](my_logs/) folder. This folder contains all metadata generated during the training process, which includes the following but not limited to:

* Hyperparameter of various trials models.
* Correlation matrix for the imported data.
* Data types and data structure, i.e. int64 or float64 & etc.
* Statistical properties of numerical variables include the mean, median, variance and quantile.
* The cardinality of categorical data.
* Ranking distribution of the target variable.
* Information regarding null value.
* Data used for plotting when comparing models during evaluation.

The metadata will not be uploaded to Git as a standard process unless specified in the `.gitignore`. Last but not least, the output model that are ready for production are kept in the [scoring](scoring/) folder. This folder contained all production unit:
* All Implemented MOJO (Model Object, Optimized)
* An Earnix postman script
* A python script that mimics the HTTP request protocol
* A predict module


[back to top](#TBC)
<br>
<br>

<a name="section-2"></a>
## PART II - Modules
<a name="evaluation"></a>
## Evaluation.py
The evaluation module is designed to compare multiple classification models concurrently.
Target that **equals** 1 is the negative class, and target **not equal** to 1 is the positive class.  
**Note:** The training data must follow this principle for any binary model, such that the confusion matrix is displayed as follows:

```
                     predicted
                      1     0  
Actual 1 (negative) [[TN   FP]
       0 (positive)  [FN    TP]]
```

For a multinomial model, please avoid using `label=0` to ensure the confusion matrix is ordered from left to right in ascending order. 
```
              predicted
           1     2     3   
Actual 1 [[11502 29    12  ]
       2  [2     5092  23  ]
       3  [8     4     2826]]
```
Such that `label!=1` can be combined on the RHS once converted to binary.

```
             predicted
             1      !=1  
Actual 1   [[11502  41]
       !=1  [10     7945]]
```


**1. To initialise the class:**
```
from Evaluation import model_eval
metric = model_eval(models, hf_test)
```
`models` take a dictionary with key-value pair, e.g.
```
{'model_id_1': h2o_model_object1,
 'model_id_2': h2o_model_object2}
 ```
`hf_test` takes any h2oFrame that must contain both predictive attributes and the target variable.

**2. To retrieve predicted results:**
```
metric.y_pred['model_id']
```

**3. To retrieve probability of positive class (!=1):**
```
metric.y_proba['model_id']
```
**4. To retrieve probability of class-1:**
```
metric.y_proba_1['model_id']
```
**5. To compare all models**
```
metric.models_stats
```
The output includes:
- ROC curve of all Models.
- The Precision and Recall curve
- Confusion Matrix
- ROC_AUC score
- Accuracy Score
- Precision & Recall

**Other class methods which can be used directly are:**
1. `plot_precision_recall_vs_threshold(precisions, recalls, thresholds)`
2. `plot_roc_curve(fpr, tpr)`

[back to top](#TBC)

<br>
<br>

<a name="data-explorer"></a>
## Data_Explorer.py
The Data_Explorer module is used to generate metadata of any pandas data frame. The metadata is stored in the `my_logs` directory when executed.

To initialise the class:
```
from Data_Explorer import data_explore
de = data_explore(df)
```


1. `de.df_info(df)` <br>
Print a text file containing all columns of the data frame, Non-null Count and Dtype of each column.

2. `de.dtypes_count(df)` <br>
Print a text file containing a summary of data type count. 

3. `de.stats(df)` <br>
Export an excel file containing mean, standard deviation and quantile information of all numerical attributes.

4. `de.cardinality(df)` <br>
Print a text file containing the cardinality of the categorical attributes. 

5. `de.cat_att`<br>
Allows you to return the categorical attributes of the data frame.

6. `de.num_att` <br>
Allows you to return the numerical attributes of the data frame.

**Other class methods which can be used directly are:**<br>
1. `de.cor_matrix(df)`  <br>
Allows you to print the correlation matrix of all numerical variables to an excel file.

[back to top](#TBC)
<br>
<br>

<a name="section-3"></a>
## PART III - Notebooks
<a name="train"></a>
### train.ipynb

This notebook synthesises the process of retraining the Oceania ranking model. You may find many of the R&D ideas helpful in the `R&D.ipynb` file.


<a name="RnD"></a>
### R&D.ipynb
This notebook contained the methodology used during the research and development. It contained many ideas that did not made it to production, e.g. multinomial classification technique, data exploration and the process of removing correlated variables. A total of 609 variables from XMLLOG are investigated during the feature selection process. [Att.xlsx](ATT.xlsx) is the universe of variables available and investigated.

[back to top](#TBC)








 






