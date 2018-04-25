# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2017/10/13
# Company : Maxent
# Email: chao.xu@maxent-inc.com

rf_base_params = {
    'criterion' : 'gini',
    'n_jobs':-1,
    'random_state' : 42,
}


rf_test_params = {
    'class_weight':map(lambda x : {0:1,1:x}, [1,5,10,100,500,1000]),
    'max_depth':range(4,50,2),
    'n_estimators':range(1,21,1),
    'max_leaf_nodes':range(2,20,2),
    'min_samples_leaf':range(50,150,10),
    'min_samples_split':range(50,150,10),
    'max_features':['sqrt','log2',None],
    'min_impurity_decrease':[i/100.0 for i in range(0,5)]
}