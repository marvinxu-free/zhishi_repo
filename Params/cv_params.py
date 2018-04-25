# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2017/10/13
# Company : Maxent
# Email: chao.xu@maxent-inc.com

cv_params = {
    'pre_dispatch': '2*n_jobs',
    'refit':True,
    'scoring':'roc_auc',
    # 'scoring':'f1_macro',
    # 'scoring':'average_precision',
    # 'scoring':'accuracy',
#     'scoring':'recall',
    'cv':5,
    "iid":False,
    'verbose':0
}

