import matplotlib.pyplot as plt
import pandas as pd
import os
from Pic.maxent_style import maxent_style
from Pic.mosaicplot import mosaic
from Pic.output_table_macdown import print_macdown_table
import scipy.stats as scs

@maxent_style
def makeMosaic(col1,col2,max_value=2,path=os.path.expanduser('~/Documents'),dpi=600,palette=None):
    """
    draw mosica picture for col1 and col2 in df
    :param col1: 
    :param col2: 
    :param df: `
    :return: 
    """
    maxValue = int(col1.max()) + 1
    if maxValue < max_value:
        return
    step = int(maxValue / max_value) + 1
    _range = list(range(-step, maxValue, step))
    _range.append(maxValue)
    cross = pd.crosstab(pd.cut(col1, _range), col2)
    print("chi2 check: \n",scs.chi2_contingency(cross))
    re = cross.div(cross.sum(1).astype(float), axis=0).fillna(0)
    re_plus = re+0.1
    re_plus_stack = re_plus.stack()
    print_macdown_table(re_plus_stack)
    title = "%s vs. %s" % (col1.name, col2.name)
    props = lambda key:{'color':next(palette),'alpha':0.9}
    fig,rec_re = mosaic(re_plus_stack,gap=0.001,properties=props,title=title)
    fig.canvas.set_window_title(title)
    path +='/{0}'.format(title)+'.png'
    fig.savefig(filename=path,dpi=dpi,format='png')
    plt.show(block=False)
    # plt.show()

@maxent_style
def valueAnomalyMosaic(value, anomaly,df,path,dpi=600,palette=None):
    """
    """
    maxValue = int(df[value].max())
    if maxValue > 5:
        step = maxValue // 5
        valueBins = list(range(-step, maxValue, step))
        valueBins.append(maxValue)
    else:
        valueBins = range(-2, 10, 2)
    anomalyBins = [-2, 0, 10]
    cross = pd.crosstab(pd.cut(df[value], valueBins),
                        pd.cut(df[anomaly], anomalyBins, labels=["normal", "anormal"]))
    # print "chi2 check ",scs.chi2_contingency(cross)
    print "corss is", cross
    print_macdown_table(cross)
    re = cross.div(cross.sum(1).astype(float), axis=0)
    # cross = cross.loc[(cross["normal"] != 0) | (cross["anormal"] != 0)]
    re_plus = re +0.02
    re_plus_stack = re_plus.stack()
    print "stack is",re_plus_stack
    title = "%s vs. %s" % (value,anomaly)
    props = lambda key:{'color':next(palette),'alpha':0.9}
    labelizer = lambda k:cross.loc[k]
    fig,rec_re = mosaic(re_plus_stack,labelizer=labelizer,gap=0.001,properties=props,title=title,axes_label=True)
    # fig,rec_re = mosaic(re_plus_stack,gap=0.001,properties=props,title=title,axes_label=True)
    fig.canvas.set_window_title(title)
    path +='/{0}'.format(title)+'.png'
    fig.savefig(filename=path,dpi=dpi,format='png')
    plt.show(block=False)

@maxent_style
def valueAnomalyMosaicBak(value, anomaly,df,path,dpi=600,palette=None):
    """
    """
    maxValue = int(df[value].max())
    if maxValue > 5:
        step = maxValue // 5
        valueBins = list(range(-step, maxValue, step))
        valueBins.append(maxValue)
    else:
        valueBins = range(-2, 10, 2)
    anomalyBins = [-2, 0, 10]
    cross = pd.crosstab(pd.cut(df[value], valueBins),
                        pd.cut(df[anomaly], anomalyBins, labels=["normal", "anormal"]))
    # print "chi2 check ",scs.chi2_contingency(cross)
    print_macdown_table(cross)
    re = cross.div(cross.sum(1).astype(float), axis=0)
    re_plus = re+0.02
    re_plus_stack = re_plus.stack()
    title = "%s vs. %s" % (value,anomaly)
    props = lambda key:{'color':next(palette),'alpha':0.9}
    fig,rec_re = mosaic(re_plus_stack,gap=0.001,properties=props,title=title,axes_label=True)
    fig.canvas.set_window_title(title)
    path +='/{0}'.format(title)+'.png'
    fig.savefig(filename=path,dpi=dpi,format='png')
    plt.show(block=False)
