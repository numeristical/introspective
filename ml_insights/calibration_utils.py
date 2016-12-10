"""Calibration of predicted probabilities."""
from __future__ import division

import numpy as np
import sklearn

try:
    from sklearn.model_selection import StratifiedKFold
except:
    from sklearn.cross_validation import StratifiedKFold


def _natural_cubic_spline_basis_expansion(xpts,knots):
    num_knots = len(knots)
    num_pts = len(xpts)
    outmat = np.zeros((num_pts,num_knots))
    outmat[:,0]= np.ones(num_pts)
    outmat[:,1] = xpts
    def make_func_H(k):
        def make_func_d(k):
            def func_d(x):
                denom = knots[-1] - knots[k-1]
                numer = np.maximum(x-knots[k-1],np.zeros(len(x)))**3 - np.maximum(x-knots[-1],np.zeros(len(x)))**3
                return numer/denom
            return func_d
        def func_H(x):
            d_fun_k = make_func_d(k)
            d_fun_Km1 = make_func_d(num_knots-1)
            return d_fun_k(x) -  d_fun_Km1(x)
        return func_H
    for i in range(1,num_knots-1):
        curr_H_fun = make_func_H(i)
        outmat[:,i+1] = curr_H_fun(xpts)
    return outmat


def prob_calibration_function(scorevec, truthvec,  reg_param_vec='default',
                                knots = 'all', extrapolate=True):
    """This function takes an uncalibrated set of scores and the true 0/1 values and returns a calibration function.

    This calibration function can then be applied to other scores from the same model and will return an accurate probability
    based on the data it has seen.  For best results, the calibration should be done on a separate validation set (not used
    to train the model).
    """
    from sklearn import linear_model
    from sklearn.metrics import log_loss, make_scorer

    knot_vec = np.unique(scorevec)
    X_mat = _natural_cubic_spline_basis_expansion(scorevec, knot_vec)


    if ((type(reg_param_vec)==str) and (reg_param_vec=='default')):
        reg_param_vec = 10**np.linspace(-4,10,43)
    print("Trying {} values of C between {} and {}".format(len(reg_param_vec),np.min(reg_param_vec),np.max(reg_param_vec)))
    reg = linear_model.LogisticRegressionCV(Cs=reg_param_vec, cv=5, scoring=make_scorer(log_loss,needs_proba=True, greater_is_better=False))
    reg.fit(X_mat, truthvec)
    print("Best value found C = {}".format(reg.C_))

    def calibrate_scores(new_scores):
        if (not extrapolate):
            new_scores = np.maximum(new_scores,knot_vec[0]*np.ones(len(new_scores)))
            new_scores = np.minimum(new_scores,knot_vec[-1]*np.ones(len(new_scores)))
        basis_exp = _natural_cubic_spline_basis_expansion(new_scores,knot_vec)
        outvec = reg.predict_proba(basis_exp)[:,1]
        return outvec
    return calibrate_scores

def train_and_calibrate_cv(model, X_tr, y_tr, cv=5):
    y_pred_xval = np.zeros(len(y_tr))
    skf = cross_validation.StratifiedKFold(y_tr, n_folds=cv,shuffle=True)
    i = 0;
    for train, test in skf:
        i = i+1
        print("training fold {} of {}".format(i, cv))
        X_train_xval = np.array(X_tr)[train,:]
        X_test_xval = np.array(X_tr)[test,:]
        y_train_xval = np.array(y_tr)[train]
        # We could also copy the model first and then fit it
        model_copy = clone(model)
        model_copy.fit(X_train_xval,y_train_xval)
        y_pred_xval[test]=model.predict_proba(X_test_xval)[:,1]
    print("training full model")
    model_copy = clone(model)
    model_copy.fit(X_tr,y_tr)
    print("calibrating function")
    calib_func = prob_calibration_function(y_pred_xval, y_tr)
    return model_copy, calib_func


