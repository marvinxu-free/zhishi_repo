import pandas as pd
import matplotlib.pyplot as plt
from Pic.maxent_style import maxent_style


@maxent_style
def ipSeg24Curve(ipSeg24,df,path=None,dpi=600,palette=None):
    """
    """
    cols = ["ipSeg24.1h.value", "message_timestamp", "ipSeg24.1h.anomaly.log"]
    data = df[df["ipSeg24"] == ipSeg24][cols]
    data = data.set_index("message_timestamp").sort_index()
    anormaly = data[data["ipSeg24.1h.anomaly.log"] > 0 ]
    print(anormaly)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(data["ipSeg24.1h.value"])
    ax.scatter(anormaly.index, anormaly["ipSeg24.1h.value"], color="red")
    ax.set_title(ipSeg24)
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.plot(data["ipSeg24.1h.value"])
    ax1.set_title(ipSeg24)
    ax1.scatter(anormaly.index, anormaly["ipSeg24.1h.value"], color="red")
    ax1.set_yscale("log")
    fig.canvas.set_window_title(ipSeg24)
    path +='/{0}'.format(ipSeg24)+'.png'
    fig.savefig(filename=path,dpi=dpi,format='png')
    plt.show(block=False)
