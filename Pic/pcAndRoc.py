# -*- coding: utf-8 -*-
# Project: maxent-ml
# Author: chaoxu create this file
# Time: 2017/8/30
# Company : Maxent
# Email: chao.xu@maxent-inc.com
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics

def roc_curve_cmp(fpr1, tpr1, fpr2, tpr2, fpr3, tpr3, fpr4, tpr4):
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    auc1 = metrics.auc(fpr1, tpr1)
    ax.plot(fpr1, tpr1, "k--", label='dfp_match(area = %0.4f)' % auc1)
    auc2 = metrics.auc(fpr2, tpr2)
    ax.plot(fpr2, tpr2, "b-.", label='mcid+did+dfp_match(area = %0.4f)' % auc2)
    ax.scatter(fpr3, tpr3, c='r', marker='o', label='mcid_match')
    ax.scatter(fpr4, tpr4, c='g', marker='o', label='mcid+did_match')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    ax.legend(loc='lower right', shadow=True)
    plt.title("Receiver operating characteristic")
    plt.show()

def prec_rec_curve(prec1, rec1, prec2, rec2, prec3, rec3, os):
    fig = plt.figure(figsize=(12, 8), dpi=80)
    ax = fig.add_subplot(111)
    ax.scatter(prec1, rec1, c='y', marker='o', label='mcid_match')
    ax.scatter(prec2, rec2, c='c', marker='o', label='mcid+did_match')
    ax.set_xlabel("precision")
    ax.set_ylabel("recall")
    ax.annotate('mcid, did and dfp match', (0.5, 1.02))
    ax.annotate('mcid match', (prec1+0.01, rec1+0.01))
    ax.annotate('mcid and did match', (prec2+0.01, rec2-0.01))
    ax.plot(prec3, rec3, 'b', label='mcid+did+dfp_match')
    ax.legend(loc='lower left')
    plt.title("Precision vs recall of " + os)
    plt.show()


def do_plot_prc():
    alpha_list = pickle.load(open("ios_alpha_list", "rb"))
    ios_dfp_only_prec = pickle.load(open("ios_dfp_only_prec", "rb"))
    ios_dfp_only_rec = pickle.load(open("ios_dfp_only_rec", "rb"))
    ios_dfp_did_prec = pickle.load(open("ios_dfp_did_prec", "rb"))
    ios_dfp_did_rec = pickle.load(open("ios_dfp_did_rec", "rb"))
    prc_curve_cmp(alpha_list, ios_dfp_only_prec, ios_dfp_only_rec, ios_dfp_did_prec, ios_dfp_did_rec, 'ios')
    mcid_prec = 1.0
    mcid_rec = 0.0805369127517
    mcid_did_prec = 1.0
    mcid_did_rec = 0.0805369127517
    prec_rec_curve(mcid_prec, mcid_rec, mcid_did_prec, mcid_did_rec, ios_dfp_did_prec, ios_dfp_did_rec, 'ios')
    alpha_list = pickle.load(open("android_alpha_list", "rb"))
    android_dfp_only_prec = pickle.load(open("android_dfp_only_prec", "rb"))
    android_dfp_only_rec = pickle.load(open("android_dfp_only_rec", "rb"))
    android_dfp_did_prec = pickle.load(open("android_dfp_did_prec", "rb"))
    android_dfp_did_rec = pickle.load(open("android_dfp_did_rec", "rb"))
    mcid_prec = 1.0
    mcid_rec = 0.0925492816572
    mcid_did_prec = 1.0
    mcid_did_rec = 0.0925492816572
    prc_curve_cmp(alpha_list, android_dfp_only_prec, android_dfp_only_rec, android_dfp_did_prec, android_dfp_did_rec, 'android')
    prec_rec_curve(mcid_prec, mcid_rec, mcid_did_prec, mcid_did_rec, android_dfp_did_prec, android_dfp_did_rec, 'android')

def prc_curve(alpha, prec, rec, min_prec, min_rec, max_prec, max_rec, os="android"):
    """
    This function is to plot the precision and recall corresponding to different alpha.
    Parameters
    ----------
    alpha:
    avg_prec: average precision of k-fold
    avg_rec: average recall of k-fold
    min_prec: minimal precision of k-fold
    min_rec: minimal recall of k-fold
    max_prec: maximum precision of k-fold
    max_rec: maximum recall of k-fold

    Returns
    -------
    no return
    
    """
    fig = plt.figure(figsize=(12, 8), dpi=80)
    ax = fig.add_subplot(311)
    # ax.set_title("%s" % "")
    ax.plot(alpha, prec, 'b-', label='average_precision')
    ax.plot(alpha, rec, 'r-', label='average_recall')
    ax.set_ylabel("prec/recall")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.2)
    ax.legend(loc='best')
    plt.title("Precision and recall of " + os)
    ax = fig.add_subplot(312)
    ax.plot(alpha, min_prec, 'b--', label='min_precision')
    ax.plot(alpha, min_rec, 'r--', label='min_recall')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.2)
    ax.set_ylabel("prec/recall")
    ax.legend(loc='best')
    ax = fig.add_subplot(313)
    ax.plot(alpha, max_prec, 'b+', label='max_precision')
    ax.plot(alpha, max_rec, 'r+', label='max_recall')
    ax.set_xlabel("alpha")
    ax.set_ylabel("prec/recall")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.2)
    ax.legend(loc='best')
    plt.show()


def prc_curve_with_confid(alpha, prec, rec, min_prec, min_rec, max_prec, max_rec, os="android"):
    """
    This function is to plot the precision and recall corresponding to different alpha.
    Parameters
    ----------
    alpha:
    avg_prec: average precision of k-fold
    avg_rec: average recall of k-fold
    min_prec: minimal precision of k-fold
    min_rec: minimal recall of k-fold
    max_prec: maximum precision of k-fold
    max_rec: maximum recall of k-fold

    Returns
    -------
    no return

    """
    fig = plt.figure(figsize=(12, 8), dpi=80)
    ax = fig.add_subplot(111)
    ax.plot(alpha, prec, 'b-', label='precision')
    ax.plot(alpha, rec, 'r-', label='recall')
    ax.plot(alpha, min_prec, 'b--', label='min_precision')
    ax.plot(alpha, min_rec, 'r--', label='min_recall')
    ax.plot(alpha, max_prec, 'b+', label='max_precision')
    ax.plot(alpha, max_rec, 'r+', label='max_recall')
    ax.set_ylabel("prec/recall")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.0, 1.2)
    ax.legend(loc='best')
    plt.title("Precision and recall of " + os)
    plt.show()

def do_plot_roc():
    tpr1 = pickle.load(open("android_dfp_only_tpr", "rb"))
    print tpr1
    fpr1 = pickle.load(open("android_dfp_only_fpr", "rb"))
    print fpr1
    tpr2 = pickle.load(open("android_mcid_did_dfp_tpr", "rb"))
    print tpr2
    fpr2 = pickle.load(open("android_mcid_did_dfp_fpr", "rb"))
    print fpr2
    tpr3 = 0.0925492816572
    fpr3 = 0.0
    tpr4 = 0.0925492816572
    fpr4 = 0.0
    roc_curve_cmp(fpr1, tpr1, fpr2, tpr2, fpr3, tpr3, fpr4, tpr4)

def prc_curve_cmp(alpha, prec1, rec1, prec2, rec2, os):
    fig = plt.figure(figsize=(12, 8), dpi=80)
    ax = fig.add_subplot(111)
    ax.plot(alpha, prec1, 'b', label='dfp_precision')
    ax.plot(alpha, rec1, 'y', label='dfp_recall')
    ax.plot(alpha, prec2, 'r', label='mcid+did+dfp_precision')
    ax.plot(alpha, rec2, 'c', label='mcid+did+dfp_recall')
    ax.set_ylabel("prec/recall")
    ax.set_xlabel("alpha")
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.6, 1.2)
    ax.legend(loc='best')
    plt.title("Precision and recall of " + os)
    plt.show()


def visualizeRoc(fpr, tpr, auc):
    """
    This function is to plot the roc curve.

    Parameters
    ----------
    fpr: false positive rate corresponding to different alpha
    tpr: true positive rate corresponding to different alpha
    auc: area under curve

    Returns
    -------
    no return

    """
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr)
    ax.set_title("auc:%f" % auc)
    ax.fill_between(fpr, tpr)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    plt.show()