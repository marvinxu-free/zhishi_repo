# -*- coding: utf-8 -*-
# Project: local-spark
# Author: chaoxu create this file
# Time: 2017/8/15
# Company : Maxent
# Email: chao.xu@maxent-inc.com
import inspect

import matplotlib.pyplot as plt

from Pic.maxent_style import maxent_style
from Utils.common.patterns import fraud_match

__path = inspect.stack()[0][1]


@maxent_style
def scoreVsTime(df, path=__path, dpi=600, title="score vs time", palette=None):
    df = df[["timestamp", "score"]]
    df = df.set_index("timestamp")
    df_1 = df.where(df["score"] <= 70).resample("H").count()
    df_2 = df.where((df["score"] > 70) & (df["score"] <= 90)).resample("H").count()
    df_3 = df.where(df["score"] > 90).resample("H").count()
    fig, axes = plt.subplots()
    axes.plot_date(df_1.index, df_1.values, color=next(palette), fmt="-", label="normal")
    axes.plot_date(df_2.index, df_2.values, color=next(palette), fmt="-", label="suspect")
    axes.plot_date(df_3.index, df_3.values, color=next(palette), fmt="-", label="fraud")
    axes.legend(loc='upper right')
    axes.set_title(title)
    path += '/{0}'.format(title) + '.png'
    fig.savefig(filename=path, dpi=dpi, format='png')
    plt.show(block=False)


@maxent_style
def fraudType(df, path=__path, dpi=600, title="fraud type", \
              _type="normal", palette=None):
    fraud_cols = df.columns.values[fraud_match(df.columns.values)]
    print fraud_cols
    df = df.set_index("timestamp")
    if _type == "normal":
        df_1 = df.where(df["score"] <= 70)
    elif _type == "suspect":
        df_1 = df.where((df["score"] > 70) & (df["score"] <= 90))
    elif _type == "fraud":
        df_1 = df.where(df["score"] > 90)

    fig, axes = plt.subplots()
    for col in fraud_cols:
        df_1 = df_1[col].resample("H").count()
        axes.plot_date(df_1.index, df_1.values, color=next(palette), fmt="-", label=col)
    axes.legend(loc='upper right')
    axes.set_title(title)
    path += '/{0}'.format(title) + '.png'
    fig.savefig(filename=path, dpi=dpi, format='png')
    plt.show(block=False)


@maxent_style
def eventType(df, path=__path, dpi=600, title="event type", \
              _type="normal", palette=None):
    df = df.set_index("timestamp")
    if _type == "normal":
        df_1 = df.loc[df["score"] <= 70]
    elif _type == "suspect":
        df_1 = df.loc[(df["score"] > 70) & (df["score"] <= 90)]
    else:
        df_1 = df.loc[df["score"] > 90]

    df_Transaction = df_1.where(df_1["event_type"] == "Transaction").resample("H").count()
    df_Act = df_1.where(df_1["event_type"] == "CreateAccount").resample("H").count()
    fig, axes = plt.subplots()
    axes.plot_date(df_Transaction.index.values, df_Transaction["event_type"].values, color=next(palette), fmt="-",
                   label="Transaction")
    axes.plot_date(df_Act.index.values, df_Act["event_type"].values, color=next(palette), fmt="-",
                   label="CreateAccount")
    axes.legend(loc='upper right')
    axes.set_title(title)
    path += '/{0}'.format(title) + '.png'
    fig.savefig(filename=path, dpi=dpi, format='png')
    plt.show(block=False)
