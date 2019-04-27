import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from sklearn.metrics import precision_recall_curve

def plot_pr_curve(truth_vec, score_vec, x_axis='precision', **kwargs):
    prec, rec, _ = precision_recall_curve(truth_vec,score_vec)
    if x_axis=='precision':
        plt.plot(prec[:-1], rec[:-1], **kwargs)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
    else:
        plt.plot(rec[:-1], prec[:-1], **kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
    plt.xlim([0,1])
    plt.ylim([0,1])

def plot_pr_curves(truth_vec_list, score_vec_list, x_axis='precision', **kwargs):
    for i in range(len(truth_vec_list)):
        prec, rec, _ = precision_recall_curve(truth_vec_list[i],score_vec_list[i])
        if x_axis=='precision':
            plt.plot(prec[:-1], rec[:-1], **kwargs)
            plt.xlabel('Precision')
            plt.ylabel('Recall')
        else:
            plt.plot(rec[:-1], prec[:-1], **kwargs)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
        plt.xlim([0,1])
        plt.ylim([0,1])


def histogram_pair(value_vec, binary_vec, bins, smoothing_const=.01, prior_prob=.5, rel_risk=False, 
                    error_bar_alpha=.05,  figsize = (12,6), **kwargs):
    """This is a tool to explore the relationship between a numerical feature and a 1/0 binary outcome.

    Author: Brian Lucena

    It plots two histograms: one is of the values of the feature when the binary outcome is positive (1)
    and the other when it is negative (0).

    It then gives the marginal empirical probability of being a 1 given that the numerical feature
    is in a particular value range.

    In practice, it often takes some experimentation to find the appropriate bin endpoints for a
    particular feature.

    If the data contains 'NaN' values, it will also draw two small horizontal (dotted and dashed)
    lines, indicating the probabilities given NaN and not NaN respectively.
    """
    nan_mask = np.isnan(value_vec)
    num_nans = np.sum(nan_mask)
    if num_nans > 0:
        nan_binary_vec = binary_vec[nan_mask]
        binary_vec = binary_vec[~nan_mask]
        value_vec = value_vec[~nan_mask]
        nan_avg_value = np.mean(nan_binary_vec)
        reg_avg_value = np.mean(binary_vec)
    # digitized_value_vec = np.digitize(value_vec, bins)
    # x_pts_to_graph = np.array([np.mean(value_vec[digitized_value_vec==i]) for i in np.unique(digitized_value_vec)])
    # print(x_pts_to_graph)
    out0 = plt.hist(value_vec[binary_vec == 0], bins=bins, **kwargs)
    out1 = plt.hist(value_vec[binary_vec == 1], bins=bins, **kwargs)
    plt.close()
    plt.figure(figsize=figsize)
    plt.subplot(2, 1, 1)
    plt.hist((value_vec[binary_vec == 0],value_vec[binary_vec == 1]), stacked=True, bins=bins, **kwargs)
    bin_leftpts = (out1[1])[:-1]
    bin_rightpts = (out1[1])[1:]
    default_bin_centers = (bin_leftpts + bin_rightpts) / 2
    digitized_value_vec = np.digitize(value_vec, bins)
    bin_centers = np.array([np.mean(value_vec[digitized_value_vec==i]) if i in np.unique(digitized_value_vec) else default_bin_centers[i-1] for i in np.arange(len(bins)-1)+1])
    prob_numer = out1[0]
    prob_denom = out1[0] + out0[0]
    smoothing_const = .001
    probs = (prob_numer + prior_prob * smoothing_const) / (prob_denom + smoothing_const)
    # print(bin_centers)
    # print(probs)
    plt.subplot(2, 1, 2)
    if rel_risk:
        plt.plot(bin_centers, np.log10(probs / prior_prob))
        # plt.errorbar(bin_centers, probs, yerr=1.96 * probs * (1 - probs) / np.sqrt(prob_denom), capsize=3)
        plt.xlim(bin_leftpts[0], bin_rightpts[-1])
    else:
        plt.plot(bin_centers[:len(probs)], probs)
        plt.xlim(bin_leftpts[0], bin_rightpts[-1])
        yerr_mat_temp = beta.interval(1-error_bar_alpha,out1[0]+1,out0[0]+1)
        yerr_mat = np.vstack((yerr_mat_temp[0],yerr_mat_temp[1])) - probs
        yerr_mat[0,:] = -yerr_mat[0,:]
        plt.errorbar(bin_centers[:len(probs)], probs, yerr=yerr_mat, capsize=5)
        plt.xlim(bin_leftpts[0], bin_rightpts[-1])
        if num_nans > 0:
            plt.hlines(y=nan_avg_value, xmin=bin_leftpts[0], xmax=bin_leftpts[1], linestyle='dotted')
            plt.hlines(y=reg_avg_value, xmin=bin_leftpts[0], xmax=bin_leftpts[1], linestyle='dashed')
    return {'bin_centers': bin_centers, 'probs': probs, 'prob_numer': prob_numer, 'prob_denom': prob_denom}




