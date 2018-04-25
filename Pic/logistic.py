# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2017/9/7
# Company : Maxent
# Email: chao.xu@maxent-inc.com
import  pandas as pd
import matplotlib.pylab as plt
from Pic.maxent_style import *
import numpy as np

def model(x,clf):
    X_new = np.dot(x, clf.coef_.T) + clf.intercept_
    return X_new, 1 / (1 + np.exp(-X_new))

@maxent_style
def logitCure(clf,X,y,palette=None):
    X_new, loss = model(X, clf=clf)
    loss = loss.ravel()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y_new = y.reset_index(drop=True)
    X_0 = X_new[y_new[y_new.label == 0].index.values]
    loss_0 = loss[y_new[y_new.label == 0].index.values]
    X_1 = X_new[y_new[y_new.label == 1].index.values]
    loss_1 = loss[y_new[y_new.label == 1].index.values]
    ax.scatter(X_0, loss_0, color=next(palette), label="label 0")
    ax.scatter(X_1, loss_1, color=next(palette), label="label 1")
    patches, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(patches, labels, loc='best', \
                    fontsize='x-small', handlelength=0.5, handletextpad=0.8)
    plt.show()