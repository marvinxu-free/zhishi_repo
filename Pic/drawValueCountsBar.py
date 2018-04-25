# -*- coding: utf-8 -*-
# Project: maxent-ml
# Author: chaoxu create this file
# Time: 2017/7/11
# Company : Maxent
# Email: chao.xu@maxent-inc.com

"""
this file is used to draw data cloumns which has mutiple values as bar
"""
import traceback
import matplotlib.pyplot as plt
import pandas as pd
from Pic.maxent_style import maxent_style,remove_palette

@maxent_style
@remove_palette
def valueCountBar(df,col=None,title=None,xlabel=None,ylabel=None):
    try:
        if col == None:
            print("you should give column names")
        else:
            count_classes = pd.value_counts(df[col], sort = True).sort_index()
            count_classes.plot(kind = 'bar')
            # plt.title("Fraud class histogram")
            # plt.xlabel("Class")
            # plt.ylabel("Frequency")
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show(block=True)
    except Exception as e:
        print(e)
        print(traceback.format_exc())


