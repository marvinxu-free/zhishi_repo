# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2017/10/13
# Company : Maxent
# Email: chao.xu@maxent-inc.com

xgb_base_params = {
    'objective' : 'binary:logistic',
#     'objective' : 'binary:logitraw',
    'nthread':4,
#     'scale_pos_weight':scale_ios_ratio,
#     'missing':-6.666,
    'seed':42
}

xgb_test_params = {
    'learning_rate' : [0.05,0.1,0.5],
    'n_estimators':range(10,200,10),
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2),
    'gamma':[i/10.0 for i in range(0,5)],
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)],
    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05],
}