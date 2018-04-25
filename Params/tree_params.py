# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2017/10/13
# Company : Maxent
# Email: chao.xu@maxent-inc.com
"""
this file is used to give some params for decision tree
"""

tree_base_params = {
    'criterion' : 'gini',
    'random_state' : 42,
}

tree_test_params = {
    'class_weight':map(lambda x : {0:1,1:x}, [1,5,10,100,500,1000]),
    'max_depth':range(4,20,2),
    'max_leaf_nodes':range(2,20,2),
    'min_samples_leaf':range(30,100,10),
    'min_samples_split':range(30,100,10),
    'min_impurity_decrease':[i/100.0 for i in range(0,5)]
}
