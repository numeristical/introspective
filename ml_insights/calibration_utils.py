"""Calibration of predicted probabilities."""
from __future__ import division

import numpy as np
import sklearn
import random
import matplotlib.pyplot as plt
from scipy.stats import binom

try:
    from sklearn.model_selection import StratifiedKFold, KFold
except:
    from sklearn.cross_validation import StratifiedKFold, KFold

from .utils import _gca


def _natural_cubic_spline_basis_expansion(xpts, knots):
    num_knots = len(knots)
    num_pts = len(xpts)
    outmat = np.zeros((num_pts,num_knots))
    outmat[:, 0] = np.ones(num_pts)
    outmat[:, 1] = xpts

    def make_func_H(k):
        def make_func_d(k):
            def func_d(x):
                denom = knots[-1] - knots[k-1]
                numer = np.maximum(x-knots[k-1], np.zeros(len(x))) ** 3 - np.maximum(x-knots[-1], np.zeros(len(x))) ** 3
                return numer/denom
            return func_d

        def func_H(x):
            d_fun_k = make_func_d(k)
            d_fun_Km1 = make_func_d(num_knots-1)
            return d_fun_k(x) -  d_fun_Km1(x)
        return func_H
    for i in range(1, num_knots-1):
        curr_H_fun = make_func_H(i)
        outmat[:, i+1] = curr_H_fun(xpts)
    return outmat


def prob_calibration_function(truthvec, scorevec, reg_param_vec='default', knots='sample',
                              method='logistic', force_prob=True, eps=1e-15, max_knots=200,
                              transform_fn='none', random_state=942, verbose=False, cv_folds=5,
                              unity_prior_weight=1, unity_prior_gridsize=20):
    """This function takes an uncalibrated set of scores and the true 0/1 values and returns a calibration function.

    This calibration function can then be applied to other scores from the same model and will return an accurate probability
    based on the data it has seen.  For best results, the calibration should be done on a separate validation set (not used
    to train the model).

    Parameters
    ----------
    truthvec : A numpy array containing the true values that is the target of the calibration.  For binary
     classification these are typically your 0/1 values.

    scorevec : A numpy array containing the scores that are not appropriate to be used as probabilities.  These do not
     necessarily need to be between 0 and 1, though that is the typical usage.

    reg_param_vec:  The vector of C-values (if method = 'logistic') or alpha values (if method = 'ridge') that the calibration should
      search across.  If reg_param_vec = 'default' (which is the default) then it picks a reasonable set of values to search across.

    knots: Default is 'sample', which means it will randomly pick a subset of size max_knots from the unique values in scorevec (while always
        keeping the largest and smallest value).  If knots='all' it will use all unique values of scorevec as knots.  This may yield a
        better calibration, but will be slower.

    method : 'logistic' or 'ridge'
        The default is 'logistic', which is best if you plan to use log-loss as your metric.  You can also use "ridge" which may
        be better if Brier Score is your metic of interest.  However, "ridge" may be less stable and robust, especially when used
        for probabilities.

    force_prob: This is ignored for method = 'logistic'.  For method = 'ridge', if set to True (the default), it will ensure that the
        values coming out of the calibration are between eps and 1-eps.

    eps: default is 1e-15.  Applies only if force_prob = True and method = 'ridge'.  See force_prob above.

    max_knots:  The number of knots to use when knots='sample'.  See knots above.

    random_state: default is 942 (a particular value to ensure consistency when running multiple times).  User can supply a different value
        if they want a different random seed.

    Returns
    ---------------------

    A function object which takes a numpy array (or a single number) and returns the output of the calculated calibration function.

    """
    from sklearn import linear_model
    from sklearn.metrics import log_loss, make_scorer

    if (unity_prior_weight>0):
        scorevec_coda, truthvec_coda = create_yeqx_bias_vectors(unity_prior_gridsize)
        coda_wt = unity_prior_weight/unity_prior_gridsize
        weightvec = np.concatenate((np.ones(len(scorevec)), coda_wt * np.ones(len(scorevec_coda))))
        scorevec = np.concatenate((scorevec, scorevec_coda))
        truthvec = np.concatenate((truthvec, truthvec_coda))

    if transform_fn != 'none':
        scorevec = transform_fn(scorevec)

    knot_vec = np.unique(scorevec)
    if (knots == 'sample'):
        num_unique = len(knot_vec)
        if (num_unique > max_knots):
            smallest_knot, biggest_knot = knot_vec[0], knot_vec[-1]
            inter_knot_vec = knot_vec[1:-1]
            random.seed(random_state)
            random.shuffle(inter_knot_vec)
            reduced_knot_vec = inter_knot_vec[:(max_knots-2)]
            reduced_knot_vec = np.concatenate((reduced_knot_vec, [smallest_knot, biggest_knot]))
            reduced_knot_vec = np.concatenate((reduced_knot_vec, np.linspace(0, 1, 21)))
            if (unity_prior_weight>0):
                reduced_knot_vec = np.concatenate((reduced_knot_vec, scorevec_coda))
            knot_vec = np.unique(reduced_knot_vec)
        if verbose:
            print("Originally there were {} knots.  Reducing to {} while preserving first and last knot.".format(num_unique, len(knot_vec)))
    X_mat = _natural_cubic_spline_basis_expansion(scorevec, knot_vec)

    if (method == 'logistic'):
        if ((type(reg_param_vec) == str) and (reg_param_vec == 'default')):
            reg_param_vec = 10**np.linspace(-7, 5, 61)
        if verbose:
            print("Trying {} values of C between {} and {}".format(len(reg_param_vec), np.min(reg_param_vec), np.max(reg_param_vec)))
        reg = linear_model.LogisticRegressionCV(Cs=reg_param_vec, cv=StratifiedKFold(cv_folds, shuffle=True),
                                                scoring=make_scorer(log_loss, needs_proba=True, greater_is_better=False))
        if (unity_prior_weight>0):
            reg.fit(X_mat, truthvec, weightvec)
        else:
            reg.fit(X_mat, truthvec)
        if verbose:
            print("Best value found C = {}".format(reg.C_))

    if (method == 'ridge'):
        if ((type(reg_param_vec) == str) and (reg_param_vec == 'default')):
            reg_param_vec = 10**np.linspace(-7, 7, 71)
        if verbose:
            print("Trying {} values of alpha between {} and {}".format(len(reg_param_vec), np.min(reg_param_vec),np.max(reg_param_vec)))
        reg = linear_model.RidgeCV(alphas=reg_param_vec, cv=KFold(cv_folds, shuffle=True), scoring=make_scorer(mean_squared_error_trunc,needs_proba=False, greater_is_better=False))
        reg.fit(X_mat, truthvec)
        if verbose:
            print("Best value found alpha = {}".format(reg.alpha_))

    def calibrate_scores(new_scores):
        new_scores = np.maximum(new_scores,knot_vec[0]*np.ones(len(new_scores)))
        new_scores = np.minimum(new_scores,knot_vec[-1]*np.ones(len(new_scores)))
        if transform_fn != 'none':
            new_scores = transform_fn(new_scores)
        basis_exp = _natural_cubic_spline_basis_expansion(new_scores,knot_vec)
        if (method == 'logistic'):
            outvec = reg.predict_proba(basis_exp)[:,1]
        if (method == 'ridge'):
            outvec = reg.predict(basis_exp)
            if force_prob:
                outvec = np.where(outvec < eps, eps, outvec)
                outvec = np.where(outvec > 1-eps, 1-eps, outvec)
        return outvec

    return calibrate_scores


def mean_squared_error_trunc(y_true, y_pred,eps=1e-15):
    y_pred = np.where(y_pred<eps,eps,y_pred)
    y_pred = np.where(y_pred>1-eps,1-eps,y_pred)
    return np.average((y_true-y_pred)**2)


def prob_calibration_function_multiclass(truthvec, scoremat, verbose=False, **kwargs):
    """This function takes an uncalibrated set of scores and the true 0/1 values and returns a calibration function.

    This calibration function can then be applied to other scores from the same model and will return an accurate probability
    based on the data it has seen.  For best results, the calibration should be done on a separate validation set (not used
    to train the model).

    Parameters
    ----------
    truthvec : A numpy array containing the true values that is the target of the calibration.  For binary
     classification these are typically your 0/1 values.

    scorevec : A numpy array containing the scores that are not appropriate to be used as probabilities.  These do not
     necessarily need to be between 0 and 1, though that is the typical usage.

    reg_param_vec:  The vector of C-values (if method = 'logistic') or alpha values (if method = 'ridge') that the calibration should
      search across.  If reg_param_vec = 'default' (which is the default) then it picks a reasonable set of values to search across.

    knots: Default is 'sample', which means it will randomly pick a subset of size max_knots from the unique values in scorevec (while always
        keeping the largest and smallest value).  If knots='all' it will use all unique values of scorevec as knots.  This may yield a
        better calibration, but will be slower.

    method : 'logistic' or 'ridge'
        The default is 'logistic', which is best if you plan to use log-loss as your metric.  You can also use "ridge" which may
        be better if Brier Score is your metic of interest.  However, "ridge" may be less stable and robust, especially when used
        for probabilities.

    force_prob: This is ignored for method = 'logistic'.  For method = 'ridge', if set to True (the default), it will ensure that the
        values coming out of the calibration are between eps and 1-eps.

    eps: default is 1e-15.  Applies only if force_prob = True and method = 'ridge'.  See force_prob above.

    max_knots:  The number of knots to use when knots='sample'.  See knots above.

    random_state: default is 942 (a particular value to ensure consistency when running multiple times).  User can supply a different value
        if they want a different random seed.

    Returns
    ---------------------

    A function object which takes a numpy array (or a single number) and returns the output of the calculated calibration function.

    """
    from sklearn import linear_model
    from sklearn.metrics import log_loss, make_scorer

    num_classes = scoremat.shape[1]
    function_list = []
    for i in range(num_classes):
        scorevec = scoremat[:,i]
        curr_truthvec = (truthvec==i).astype(int)
        function_list.append(prob_calibration_function(curr_truthvec,scorevec,verbose=verbose,**kwargs))

    def calibrate_scores_multiclass(new_scoremat):
        a,b = new_scoremat.shape
        pre_probmat = np.zeros((a,b))
        for i in range(num_classes):
            pre_probmat[:,i] = function_list[i](new_scoremat[:,i])
        probmat = (pre_probmat.T/np.sum(pre_probmat,axis=1)).T
        #if (not extrapolate):
        #    new_scores = np.maximum(new_scores,knot_vec[0]*np.ones(len(new_scores)))
        #    new_scores = np.minimum(new_scores,knot_vec[-1]*np.ones(len(new_scores)))
        return probmat
    return calibrate_scores_multiclass, function_list

def plot_prob_calibration(calib_fn, show_baseline=True, ax=None, **kwargs):
    if ax is None:
        ax = _gca()
        fig = ax.get_figure()
    ax.plot(np.linspace(0,1,100),calib_fn(np.linspace(0,1,100)),**kwargs)
    if show_baseline:
        ax.plot(np.linspace(0,1,100),(np.linspace(0,1,100)),'k--')
    ax.axis([-0.1,1.1,-0.1,1.1])

def plot_reliability_diagram(y,x,bins=np.linspace(0,1,21),size_points=False, show_baseline=True,
                                error_bars=True, error_bar_alpha = .05, marker='+',c='red', **kwargs):
    # if ax is None:
    #     ax = plt.gca()
    #     fig = ax.get_figure()
    digitized_x = np.digitize(x, bins)
    mean_count_array = np.array([[np.mean(y[digitized_x == i]),len(y[digitized_x == i]),np.mean(x[digitized_x==i])] for i in np.unique(digitized_x)])
    x_pts_to_graph = mean_count_array[:,2]
    y_pts_to_graph = mean_count_array[:,0]
    pt_sizes = mean_count_array[:,1]
    plt.subplot(1,2,1)

    if show_baseline:
        plt.plot(np.linspace(0,1,100),(np.linspace(0,1,100)),'k--')
#        ax.loglog(np.linspace(0,1,100),(np.linspace(0,1,100)),'k--')
    for i in range(len(y_pts_to_graph)):
        if size_points:
            plt.scatter(x_pts_to_graph,y_pts_to_graph,s=pt_sizes,marker=marker,c=c, **kwargs)
        else:
            plt.scatter(x_pts_to_graph,y_pts_to_graph, c=c, **kwargs)
    plt.axis([-0.1,1.1,-0.1,1.1])
    
    if error_bars:
        yerr_mat = binom.interval(1-error_bar_alpha,pt_sizes,x_pts_to_graph)/pt_sizes - x_pts_to_graph
        yerr_mat[0,:] = -yerr_mat[0,:]
        plt.errorbar(x_pts_to_graph, x_pts_to_graph, yerr=yerr_mat, capsize=5)
    plt.subplot(1,2,2)
    plt.hist(x,bins=bins)

    return(x_pts_to_graph,y_pts_to_graph,pt_sizes)

def compact_logit(x, eps=.00001):
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
        return np.nansum(((x<=eps)*x, (x>=(1-eps))*x, ((x>eps)&(x<(1-eps)))*((1-2*eps)*(np.log(x/(1-x)))/(2*np.log((1-eps)/eps))+.5)),axis=0)

def inverse_compact_logit(x, eps=.00001):
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in log")
        warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
        return np.nansum(((x<=eps)*x, (x>=(1-eps))*x,
                          ((x>eps)&(x<(1-eps)))*
                          (1/(1+np.exp(-(x-.5)*((2/(1-2*eps))*np.log((1-eps)/eps)))))),axis=0)


def create_yeqx_bias_vectors(gridsize=10):
    scorevec_coda = np.sort(np.tile(np.arange(gridsize + 1)/gridsize, reps = (gridsize)))
    truthvec_coda = np.array([])
    for i in range(gridsize + 1):
        added_bit = np.concatenate((np.zeros(gridsize - i), np.ones(i)))
        truthvec_coda = np.concatenate((truthvec_coda, added_bit))
    return scorevec_coda, truthvec_coda
