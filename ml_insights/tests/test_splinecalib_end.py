import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import ml_insights as mli
from sklearn.metrics import roc_auc_score, log_loss

data_path = op.join(mli.__path__[0], 'data')


def test_identity_calibration():
    npts = 5000
    np.random.seed(42)
    xvec = np.random.uniform(size =npts)
    yvec = np.random.binomial(n=1, p=xvec)
    sc = mli.SplineCalib()
    sc.fit(xvec, yvec)
    tvec = np.linspace(.001,.999,999)
    max_err = np.max(np.abs(sc.calibrate(tvec) - tvec))
    assert(max_err<.015)

def test_identity_calibration_unity():
    npts = 1000
    np.random.seed(42)
    xvec = np.random.uniform(size =npts)
    yvec = np.random.binomial(n=1, p=xvec)
    sc = mli.SplineCalib(unity_prior=True, unity_prior_weight=2000, unity_prior_gridsize=200)
    sc.fit(xvec, yvec)
    tvec = np.linspace(.001,.999,999)
    max_err = np.max(np.abs(sc.calibrate(tvec) - tvec))
    assert(max_err<.01)

def test_identity_calibration_reg_param():
    npts = 5000
    np.random.seed(42)
    xvec = np.random.uniform(size =npts)
    yvec = np.random.binomial(n=1, p=xvec)
    sc = mli.SplineCalib(reg_param_vec=np.logspace(-2,2,41))
    sc.fit(xvec, yvec)
    tvec = np.linspace(.001,.999,999)
    max_err = np.max(np.abs(sc.calibrate(tvec) - tvec))
    assert(max_err<.015)

def test_mnist_calib():
    """
    This tests a multiclass calibration on data derived from MNIST 
    (using just the digits 0-4). We test the default settings and 
    ensure that the resulting log-loss gives good performance

    """
    calib_set = pd.read_csv(op.join(data_path,'mnist4_calib_set.csv'))
    test_set = pd.read_csv(op.join(data_path,'mnist4_test_set.csv'))

    preds_calib_set = calib_set.iloc[:,:-1].to_numpy()
    y_calib_set = calib_set.iloc[:,-1].to_numpy()
    preds_test = test_set.iloc[:,:-1].to_numpy()
    y_test = test_set.iloc[:,-1].to_numpy()
    sc = mli.SplineCalib()
    sc.fit(preds_calib_set, y_calib_set)
    preds_test_calibrated = sc.calibrate(preds_test)
    ll_calib = log_loss(y_test, preds_test_calibrated)
    assert(ll_calib<.2334)


