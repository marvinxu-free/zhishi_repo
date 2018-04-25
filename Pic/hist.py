# -*- coding: utf-8 -*-
from __future__ import print_function, division
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Pic.maxent_style import maxent_style, remove_palette
from Pic.maxent_font import tick_font
from itertools import combinations
import seaborn as sns
from scipy.stats import norm, t, f, betaprime, logistic, exponpow, foldnorm, poisson, zipf
from scipy.optimize import curve_fit
from scipy.misc import factorial
import re


def poisson_fit(k, lamb):
    return (lamb ** k / factorial(k)) * np.exp(-lamb)


fit_functions = [norm, t, f, foldnorm, logistic, betaprime, exponpow]
fit_names = ["norm", "t", "f", "foldnorm", "logistic", "betaprime", "exponpow"]


# @remove_palette
@maxent_style
def makeHist(col, df, dpi=600, title=None, path=None, palette=None, fit=False):
    """
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    maxValue = int(df[col].max())
    if maxValue > 10:
        step = int(maxValue / 10)
        bins = range(-step, maxValue, step)
        bins.append(maxValue)
    else:
        bins = range(-2, 10, 2)
    re_cols = pd.cut(df[col], bins)
    # print('re_cols\n', re_cols)
    re = re_cols.value_counts(sort=False)
    print('re\n', re)
    re_div = re.div(re.sum())
    print('re_idv\n', re_div)
    re_div.plot.bar(ax=ax)
    ax1 = fig.add_subplot(1, 2, 2)
    re.plot.bar(ax=ax1, logy=True)

    if title:
        ax.set_title(title)
        ax1.set_title(title + "/log")
    else:
        ax.set_title(col)
        ax1.set_title(col + "/log")
    # ax1.set_yscale("log")
    fig.canvas.set_window_title(col)
    if ".png" not in path:
        path += '/{0}'.format(col) + '.png'
    fig.savefig(filename=path, dpi=dpi, format='png')
    plt.show(block=False)


@maxent_style
@remove_palette
# def anormalyScoreHist(cols,score,df,dpi=600,path=None,bins = [-1,0,70,90,100],name="anormalyScore"):
def anormalyScoreHist(cols, score, df, dpi=600, path=None, name="anormalyScore"):
    """
    """
    maxValue = int(df[score].max())
    if maxValue > 10:
        step = int(maxValue / 10)
        bins = range(-step, maxValue, step)
        bins.append(maxValue)
    else:
        bins = range(-2, 10, 2)
    # re = pd.cut(df[cols], bins).value_counts(sort=False)
    re = pd.cut(df[score], bins=bins)
    # print('re\n',re.value_counts())
    re_group = df.groupby(re)[cols].agg('sum')
    indexs = re_group.index
    print('re_group\n', re_group)
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    re_group.plot.bar(ax=ax, stacked=True, legend=False)
    # ax.set_title("total view")
    ax.xaxis.label.set_visible(False)
    tick_font(ax, font_size="x-small", rotation=90)
    # re_group[cols] =re_group[cols].apply(lambda x: 0 if x.any() <=0 or x.any() == None else np.log(x))
    patches, labels = ax.get_legend_handles_labels()
    ax1 = fig.add_subplot(2, 1, 2)
    ax1.axis("off")
    lgd = ax1.legend(patches, labels, loc='center', ncol=4, bbox_to_anchor=(0.5, 0.2), \
                     fontsize='x-small', handlelength=0.5, handletextpad=0.8)
    fig.canvas.set_window_title(name)
    path_1 = path + '/{0}'.format(name + " total view") + '.png'
    # fig.savefig(filename=path,dpi=dpi,format='png',bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig(filename=path_1, dpi=dpi, format='png')

    for k, v in enumerate(indexs):
        df = re_group.loc[v, :]
        x_labels = df.index.values
        bar_values = df._values
        fig1 = plt.figure()
        axs = fig1.add_subplot(1, 1, 1)
        for x, y in enumerate(x_labels):
            axs.bar(x, bar_values[x])
        axs.set_xticks(range(len(x_labels)))
        axs.set_xticklabels(x_labels)
        tick_font(axs, font_size="x-small", rotation=90)
        path_2 = path + '/{0}_{1}-{2}.png'.format(name, v.left, v.right)
        # axs.set_title(v + " score distribute")
        fig1.subplots_adjust(bottom=0.4)
        fig1.canvas.set_window_title(v)
        fig1.savefig(filename=path_2, dpi=dpi, format='png')
    # plt.show(block=True)
    plt.show(block=False)


@maxent_style
@remove_palette
def dataCorr(cols, df, dpi=600, title='data correlations', path=None, filter_value=1.0):
    corr_cols = combinations(cols, 2)
    frames = []
    for corr_col in corr_cols:
        # print(tabulate(df.loc[(df[corr_col[0]] != 1) | (df[corr_col[1]] != 1),corr_col].corr(),showindex="always",)
        #                tablefmt='fancy_grid',headers=corr_col)
        frames.append(df.loc[(df[corr_col[0]] != filter_value) | (df[corr_col[1]] != filter_value), corr_col].corr())

    corr_result = pd.concat(frames)
    # print(corr_result.shape ,'\n',corr_result)
    # print_table(corr_result)
    grouped_result = corr_result.groupby(corr_result.index)
    # print(grouped_result)
    agg_result = grouped_result.agg('sum')
    agg_result[agg_result >= 1] = 1
    # print(agg_result)
    # print_table(agg_result)
    # print_macdown_table(agg_result)
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    agg_result.plot.bar(ax=ax, legend=False)
    # ax.set_title("total view")
    ax.xaxis.label.set_visible(False)
    tick_font(ax, font_size="x-small", rotation=90)
    # re_group[cols] =re_group[cols].apply(lambda x: 0 if x.any() <=0 or x.any() == None else np.log(x))
    patches, labels = ax.get_legend_handles_labels()
    ax1 = fig.add_subplot(2, 1, 2)
    ax1.axis("off")
    lgd = ax1.legend(patches, labels, loc='center', ncol=4, bbox_to_anchor=(0.5, 0.2), \
                     fontsize='x-small', handlelength=0.5, handletextpad=0.8)
    fig.canvas.set_window_title("col correlation")
    path += '/{0}'.format(title) + '.png'
    fig.savefig(filename=path, dpi=dpi, format='png')
    # plt.show()
    plt.show(block=False)


@maxent_style
def makeFeatureHist(col, col1, df, feature="ipGeo", scope=[6, 8], dpi=600, path=None, palette=None):
    """
    """
    df1 = df.loc[(df[col1] > scope[0]) & (df[col1] <= scope[1])]
    df1_city = pd.value_counts(df1[feature])
    if df1_city.size <= 0:
        return
    max_feature = df1_city.idxmax()
    df = df.loc[df[feature] == max_feature]
    df = df[col].dropna()
    for fit_name, fit_func in zip(fit_names, fit_functions):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)
        sns.distplot(ax=ax, a=df, color="blue", hist_kws={"histtype": "step", "linewidth": 3}, \
                     fit=fit_func, \
                     fit_kws={"color": next(palette), "lw": 3, "label": fit_name}, \
                     # rug=True,\
                     # rug_kws={"color": next(palette)},\
                     kde=True, \
                     kde_kws={"color": next(palette), "lw": 3, "label": "KDE"})
        ax.set_title("{0}-{1}-{2}".format(col, fit_name, max_feature))
        fig.canvas.set_window_title("{0}-{1}-{2}".format(col, fit_name, max_feature))
        path1 = path + "/{0}-{1}.png".format(col, fit_name)
        fig.savefig(filename=path1, dpi=dpi, format='png')
        # plt.show(block=True)
        plt.show(block=False)


@maxent_style
def makeSFeatureHist(col, col1, df, feature="maxentID", scope=[6, 8], dpi=600, path=None, palette=None):
    """
    """
    df = df[col].dropna()
    for fit_name, fit_func in zip(fit_names, fit_functions):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)
        sns.distplot(ax=ax, a=df, color="blue", hist_kws={"histtype": "step", "linewidth": 3}, \
                     fit=fit_func, \
                     fit_kws={"color": next(palette), "lw": 3, "label": fit_name}, \
                     # rug=True,\
                     # rug_kws={"color": next(palette)},\
                     kde=True, \
                     kde_kws={"color": next(palette), "lw": 3, "label": "KDE"})
        ax.set_title("{0}-{1}".format(col, fit_name))
        fig.canvas.set_window_title("{0}-{1}".format(col, fit_name))
        path1 = path + "/{0}-{1}.png".format(col, fit_name)
        fig.savefig(filename=path1, dpi=dpi, format='png')
        # plt.show(block=True)
        plt.show(block=False)


@maxent_style
@remove_palette
def pic_label(df, col, title, file_path, dpi=600):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax = df[col].value_counts(normalize=True).plot(kind='bar', ax=ax)
    label_0_num = df.loc[df[col] == 0].shape[0]
    label_1_num = df.loc[df[col] == 1].shape[0]
    text_print = "label = 0 num: {0}\n\nlabel = 1 num: {1}".format(label_0_num, label_1_num)
    ax.text(1.55, 0.5, text_print, ha='left', va='center')
    fig.subplots_adjust(left=0.1, right=0.7)
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right')
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha='right')
    ax.set_ylabel('ratio')
    ax.set_xlabel('label type')
    ax.set_title(title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if re.search(u'[\u4e00-\u9fff]', title):
        file_path += u"/{0}.png".format(title)
    else:
        file_path += "/{0}.png".format(title.replace(" ", "_"))
    fig.savefig(filename=file_path, dpi=dpi, format='png')
    plt.show(block=False)


@maxent_style
def makeFeatureGroupCountHist(gcol, ccol, df, dpi=600, title=None, fname=None, palette=None, normal=True, pic=False):
    """
    first: this function used to plot df to do groupby used gcol
    send: unique count with ccol and make hist and kde with ccol
    :param gcol:
    :param ccol:
    :param df:
    :param dpi:
    :param title:
    :param fname:
    :param palette:
    :param normal: use normal data or not
    :param pic:
    :return:
    """
    if normal:
        df = df.loc[df[ccol].notnull()]
    df = df.groupby(gcol).agg({ccol: 'nunique'}).reset_index()
    # if threshold:
    #     threshold_path = path + "/{0}_{1}_threshold_ratio.png".format(gcol, ccol)
    #     threshold_title = title + "阈值调整影响"
    #     pic_line_threshold_ratio(thresholds=df[ccol].values, title=threshold_title, path=threshold_path)
    if pic:
        path1 = fname + "/each_{0}_{1}_distribution.png".format(gcol, ccol)
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)
        sns.distplot(ax=ax, a=df[ccol], color="blue", hist_kws={"histtype": "step", "linewidth": 3},
                     fit=norm,
                     fit_kws={"color": next(palette), "lw": 3, "label": "normal"},
                     kde=True,
                     kde_kws={"color": next(palette), "lw": 3, "label": "KDE"})
        ax.set_title(title)
        fig.canvas.set_window_title(title)
        fig.savefig(filename=path1, dpi=dpi, format='png')
        plt.show(block=False)
    # else:
    #     makeHist(col=ccol, df=df, path=path1, title=title)
    return df


@maxent_style
def makeFeatureGroupTimeHist(gcol, df, dpi=600, title=None, fname=None, palette=None, normal=True, pic=False):
    """
    first: this function use df to groupby gcol
    second: agg new col based  tcol with max(tcol) - min(tcol)
    :param gcol:
    :param tcol:
    :param df:
    :param dpi:
    :param title:
    :param fname: save file name
    :param palette:
    :param normal:
    :param pic:
    :return:
    """

    def get_delta(row):
        _delta = 0
        ckids = list(row['ckid'])
        time_min = list(row['timestamp_min'])
        time_max = list(row['timestamp_max'])
        for i in xrange(len(ckids) - 1):
            pre_time, next_time = time_max[i], time_min[i + 1]
            _delta += next_time - pre_time
        _del_mean = _delta / len(ckids)
        return _del_mean

    df = df.groupby(['mobile', 'ckid']).agg({
        "timestamp": ["max", 'min']})
    df.columns = ["_".join(x) for x in df.columns.ravel()]
    df = df.reset_index()
    df = df.sort_values(by="timestamp_max")
    df = df.groupby("mobile").apply(lambda x: get_delta(x)).reset_index(name='delta')
    df['delta'] = df['delta'] / 1000 / 60 / 60
    if normal:
        df = df.loc[df['delta'] > 0]
    # if threshold:
    #     if isinstance(gcol, list):
    #         threshold_path = path + "/{0}_{1}_threshold_ratio.png".format("_".join(gcol), tcol)
    #     else:
    #         threshold_path = path + "/{0}_{1}_threshold_ratio.png".format(gcol, tcol)
    #     threshold_title = title + "阈值调整影响"
    #     pic_line_threshold_ratio(thresholds=df['delta'].values, title=threshold_title, num=3, path=threshold_path)
    if pic:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist()
        sns.distplot(ax=ax, a=df['delta'], color="blue", hist_kws={"histtype": "step", "linewidth": 3},
                     fit=norm,
                     fit_kws={"color": next(palette), "lw": 3, "label": "normal"},
                     kde=True,
                     kde_kws={"color": next(palette), "lw": 3, "label": "KDE"})
        ax.set_title(title)
        fig.canvas.set_window_title(title)
        ax.set_xlabel("{0}变化数量/小时".format(gcol[-1]))
        fig.savefig(filename=fname, dpi=dpi, format='png')
        plt.show(block=False)
    # else:
    #     makeHist(col='delta',df=df, path=path1, title=title)
    return df


@maxent_style
def makeFeatureCorrTimeHist(cor_cols, df, gcol=None, resample_ratio="W", dpi=600, path=None, palette=None):
    """
    this function used to draw two or more
    :param gcol:
    :param tcol:
    :param df:
    :param resample_ratio:
    :param dpi:
    :param title:
    :param path:
    :param palette:
    :return:
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index("timestamp")
    groups = []
    if gcol:
        groups.append(gcol)
    if resample_ratio:
        groups.append(pd.TimeGrouper(resample_ratio))
    grouper = df.groupby(groups)
    df1 = pd.DataFrame()
    df1[cor_cols[0]] = grouper[cor_cols[0]].count().values
    df1[cor_cols[1]] = grouper[cor_cols[1]].count().values
    xlim = (df1[cor_cols[0]].min() * 0.8, df1[cor_cols[0]].max() * 1.1)
    ylim = (df1[cor_cols[1]].min() * 0.8, df1[cor_cols[1]].max() * 1.1)
    grid = sns.jointplot(x=cor_cols[0], y=cor_cols[1], data=df1, kind="reg", xlim=xlim, ylim=ylim,
                         color=next(palette), size=12);
    # grid.fig.suptitle("{0}与{1}相关性/周".format(cor_cols[0], cor_cols[1]))
    grid.fig.suptitle("{0}与{1}相关性/周".format(cor_cols[0], cor_cols[1]))
    path1 = path + "/{0}_{1}_correlation.png".format(cor_cols[0], cor_cols[1])
    grid.savefig(filename=path1, dpi=dpi, format='png')
    plt.show(block=False)


@maxent_style
def makeCitySwitchTimeHist(df, gcol, time_scope=[20, 60], dpi=600, path=None, palette=None):
    """
    this function used to get the ratio and number change two faste from scols
    first: do group by gcol
    second: analysis scols switch to new use time less than time_scope, get the number and ratio
    third: pic a picture, analysis the distribution and correlation
    :param scols:
    :param df:
    :param gcol:
    :param time_scope:
    :param dpi:
    :param path:
    :param palette:
    :return:
    """
    df = df.copy()
    df = df.sort_values(by="timestamp")

    def get_switch_over(row):
        switch_quick_num = 0
        province = list(row['province'])
        city = list(row['city'])
        time_stamp = list(row['timestamp'])
        for i in xrange(len(time_stamp) - 1):
            pre_province, next_province = province[i], province[i + 1]
            pre_city, next_city = city[i], city[i + 1]
            pre_time, next_time = time_stamp[i], time_stamp[i + 1]
            time_delta = (next_time - pre_time) / 1000 / 60
            if pre_province == next_province:
                if pre_city != next_city and time_delta <= time_scope[0]:
                    switch_quick_num += 1
            else:
                if pre_city != next_city and time_delta <= time_scope[1]:
                    switch_quick_num += 1
        return switch_quick_num

    df_switch = df.groupby(gcol).apply(lambda x: get_switch_over(x)).reset_index(name='switch')
    # df_switch = df_switch.loc[df_switch['switch'] > 0]

    # df_city = df.groupby(gcol).agg({'city':'count'}).reset_index()
    # df1 = pd.merge(df_switch,df_city,on=gcol)
    # df1['ratio'] = df1['switch'] / df1['city']
    # xlim = (df1['switch'].min() * 0.8, df1['switch'].max() * 1.1)
    # ylim = (df1['ratio'].min() * 0.8 , df1['ratio'].max() * 1.1)
    # grid = sns.jointplot(x='switch', y='ratio', data=df1, kind="reg", xlim=xlim, ylim=ylim,
    #                      color=next(palette), size=12);
    # grid.fig.suptitle("同一主动式ID切换过快与所占比例关系")
    # grid.set_axis_labels(xlabel="同一主动式ID切换城市过快的数量", ylabel="同一主动式ID切换城市过快的比例")
    # path1 = path + "/{0}_switch_quick_ratio_correlation.png".format(gcol)
    # grid.savefig(filename=path1,dpi=dpi, format='png')
    # plt.show(block=False)
    return df_switch


@maxent_style
def makeEventSwitchHist(df, gcol, dpi=600, path=None, palette=None):
    """
    this function used to analysis the num and ratio of event type different and ckid change, too
    first: group by gcol
    sencond: get the number and ratio while event type is not same from pre and cur and ckid is difference, too
    :param df:
    :param gcol:
    :param time_scope:
    :param dpi:
    :param path:
    :param palette:
    :return:
    """
    df = df.copy()
    df = df.sort_values(by="timestamp")

    def get_switch_event_ckid_same(row, same=True):
        switch_evnet_num = 0
        type_ = list(row['type'])
        ckid = list(row['ckid'])
        time_stamp = list(row['timestamp'])
        for i in xrange(len(time_stamp) - 1):
            pre_type, next_type = type_[i], type_[i + 1]
            pre_ckid, next_ckid = ckid[i], ckid[i + 1]
            if not same and pre_type != next_type and pre_ckid != next_ckid:
                switch_evnet_num += 1
            elif same and pre_type == next_type and pre_ckid != next_ckid:
                switch_evnet_num += 1
        return switch_evnet_num

    df_same = df.groupby([gcol, 'type']).agg({"ckid": "nunique"}).reset_index()
    df_same = df_same.loc[df_same['ckid'] >= 1]
    if not df_same.empty:
        df_same = df.merge(df_same.drop(['ckid'], axis=1), on=['mobile', 'type'])
        df_same = df_same.sort_values(by="timestamp")
        df_same = df_same.groupby(gcol).apply(lambda x: get_switch_event_ckid_same(x)).reset_index(name='same')
        # df_same = df_same.loc[df_same['same'] > 0]
    df_not_same = df.groupby(gcol).agg({"type": "nunique"}).reset_index()
    df_not_same = df_not_same.loc[df_not_same['type'] > 1]
    if not df_not_same.empty:
        df_not_same = df.merge(df_not_same.drop(['type'], axis=1), on='mobile')
        df_not_same = df_not_same.sort_values(by="timestamp")
        df_not_same = df_not_same.groupby(gcol).apply(lambda x: get_switch_event_ckid_same(x, False)).reset_index(
            name='not_same')
        # df_not_same = df_not_same.loc[df_not_same['not_same'] > 0]
    return df_same, df_not_same
    # df_dev = df.groupby(gcol).agg({'ckid':'count'}).reset_index()
    # df_dev = df_dev.loc[df_dev.ckid != 0]
    # df1 = df_same.merge(df_not_same, on=gcol).merge(df_dev, on=gcol)
    # df1['same_ratio'] = df1['same'] / df1['ckid']
    # df1['not_same_ratio'] = df1['not_same'] / df1['ckid']
    # xlim1 = (df1['same'].min() * 0.8, df1['same'].max() * 1.1)
    # ylim1 = (df1['same_ratio'].min() * 0.8 , df1['same_ratio'].max() * 1.1)
    # color_ = next(palette)
    # grid1 = sns.jointplot(x='same', y='same_ratio', data=df1, kind="reg", xlim=xlim1, ylim=ylim1,
    #                      color=color_, size=12);
    # grid1.fig.suptitle("同一设备前后事件类型相同主动式指纹ID数量\比例\分布")
    # grid1.set_axis_labels(xlabel="同一设备前后事件类型相同主动式指纹ID数量", ylabel="同一设备前后事件类型相同主动式指纹ID比例")
    # xlim2 = (df1['not_same'].min() * 0.8, df1['not_same'].max() * 1.1)
    # ylim2 = (df1['not_same_ratio'].min() * 0.8 , df1['not_same_ratio'].max() * 1.1)
    # grid2 = sns.jointplot(x='not_same', y='not_same_ratio', data=df1, kind="reg", xlim=xlim2, ylim=ylim2,
    #                      color=color_, size=12);
    # grid2.fig.suptitle("同一设备前后事件类型不同主动式指纹ID数量\比例\分布")
    # grid2.set_axis_labels(xlabel="同一设备前后事件类型不同主动式指纹ID数量", ylabel="同一设备前后事件类型不同主动式指纹ID比例")
    # path1 = path + "/same_event_ckid_{0}_switch_ratio_num_correlation.png".format(gcol)
    # path2 = path + "/not_same_event_ckid_{0}_switch_ratio_num_correlation.png".format(gcol)
    # grid1.savefig(filename=path1,dpi=dpi, format='png')
    # grid2.savefig(filename=path2,dpi=dpi, format='png')
    # plt.show(block=False)


@maxent_style
def hist_with_poisson(col, df, dpi=600, poison_r=True, title=None, fname=None, palette=None):
    """
    plot hist with poisson distribution fit curve
    :param col:
    :param df:
    :param dpi:
    :param poison_r:
    :param title:
    :param fname:
    :param palette:
    :return:
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    maxValue = int(df[col].max())
    minValue = int(df[col].min())
    if maxValue > 10:
        step = int(maxValue / 10)
        bins = range(-step, maxValue, step)
        bins.append(maxValue)
    else:
        bins = range(-1, 10, 1)
    re_cols = pd.cut(df[col], bins)
    re = re_cols.value_counts(sort=False)
    re_div = re.div(re.sum())
    br = re_div.plot.bar(ax=ax)
    ax_log = fig.add_subplot(1, 2, 2)
    br_log = re.plot.bar(ax=ax_log, logy=True)
    x_plot = np.linspace(minValue, maxValue, 1000)
    if poison_r:
        bin_middles = map(lambda patch: patch._x + patch._width / 2.0, br_log.containers[0])
        bar_heights = map(lambda x: x._height, br_log.containers[0])
        parameters, _ = curve_fit(poisson_fit, bin_middles, bar_heights)
        ax_log.plot(x_plot, poisson_fit(x_plot, *parameters), "r-", lw=2)
    else:
        bin_middles = map(lambda patch: patch._x + patch._width / 2.0, br_log.containers[0])
        bar_heights = map(lambda x: x._height, br.containers[0])
        parameters, _ = curve_fit(poisson_fit, bin_middles, bar_heights)
        ax.plot(x_plot, poisson_fit(x_plot, *parameters), "r-", lw=2)
    ax.set_title(title)
    ax_log.set_title(title + "/log")
    fig.canvas.set_window_title(col)
    fig.savefig(filename=fname, dpi=dpi, format='png')
    plt.show(block=False)


@maxent_style
def hist_with_zip(col, df, dpi=600, title=None, fname=None, palette=None):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    maxValue = int(df[col].max())
    minValue = int(df[col].min())
    if maxValue > 10:
        step = int(maxValue / 10)
        bins = range(minValue, maxValue, step)
        bins.append(maxValue)
    else:
        bins = range(minValue, 10, 1)
    re_cols = pd.cut(df[col], bins, right=False, include_lowest=True)
    re = re_cols.value_counts(sort=False)
    re_div = re.div(re.sum())
    br = re_div.plot.bar(ax=ax)
    ax_log = fig.add_subplot(1, 2, 2)
    br_log = re.plot.bar(ax=ax_log, logy=True)

    bin_middles = np.array(map(lambda patch: patch._x + patch._width / 2.0, br.containers[0]))
    bar_heights = np.array(map(lambda x: x._height, br.containers[0]))
    norm_data = bin_middles[0]
    pi = bar_heights[0]
    x_plot = np.linspace(bin_middles[0], bin_middles[-1], 1000)

    def zip_fit(k, _lamb):
        k_zero = k - norm_data
        return (k_zero == 0) * pi + \
               (1 - pi) * (_lamb ** k_zero / factorial(k_zero) * np.exp(-_lamb))

    parameters, _ = curve_fit(zip_fit, bin_middles, bar_heights)
    fit_p = zip_fit(x_plot, *parameters)
    ax.plot(x_plot, fit_p, "r-", lw=2)
    ax.set_title(title)
    ax_log.set_title(title + "/log")
    fig.canvas.set_window_title(col)
    fig.savefig(filename=fname, dpi=dpi, format='png')
    plt.show(block=False)


@maxent_style
def hist_two_fig(col, df, dpi=600, title=None, fname=None, sparse=True, palette=None):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(2, 1, 1)
    maxValue = int(df[col].max())
    minValue = int(df[col].min())
    if sparse:
        bins = range(minValue, maxValue, 1)
        bins.append(maxValue)
    else:
        if maxValue > 10:
            step = int(maxValue / 10)
            bins = range(minValue, maxValue, step)
            bins.append(maxValue)
        else:
            bins = range(minValue, 10, 1)
    re_cols = pd.cut(df[col], bins, right=False, include_lowest=True)
    re = re_cols.value_counts(sort=False)
    re_div = re.div(re.sum())
    br = re_div.plot.bar(ax=ax)
    ax_log = fig.add_subplot(2, 1, 2)
    br_log = re.plot.bar(ax=ax_log, logy=True)
    ax.set_title(title)
    ax.set_ylabel("比例")
    ax.xaxis.set_ticks([])
    # ax_log.set_title(title + "/log")
    ax_log.set_ylabel("数量/log")
    if sparse:
        bin_middles = np.array(map(lambda patch: patch._x + patch._width / 2.0, br.containers[0]))
        log_bin_middles = np.array(map(lambda patch: patch._x + patch._width / 2.0, br_log.containers[0]))
        plt.sca(ax)
        plt.xticks(bin_middles, bins, rotation=90)
        plt.sca(ax_log)
        plt.xticks(log_bin_middles, bins, rotation=90)
    fig.canvas.set_window_title(col)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    fig.savefig(filename=fname, dpi=dpi, format='png')
    plt.show(block=False)
