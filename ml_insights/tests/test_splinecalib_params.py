import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import ml_insights as mli
from sklearn.metrics import roc_auc_score, log_loss

data_path = op.join(mli.__path__[0], 'data')


def test_knot_ss():
    npts = 100
    np.random.seed(42)
    xvec = np.random.uniform(size =npts)
    yvec = np.random.binomial(n=1, p=xvec)
    sc = mli.SplineCalib(knot_sample_size=20, force_knot_endpts=True)
    sc.fit(xvec, yvec)
    t1 = len(sc.knot_vec)==20
    t2 = np.min(xvec) in sc.knot_vec
    t3 = np.max(xvec) in sc.knot_vec
    assert(t1 and t2 and t3)

def test_add_knots():
    npts = 5000
    np.random.seed(42)
    xvec = np.random.uniform(size =npts)
    yvec = np.random.binomial(n=1, p=xvec)
    sc = mli.SplineCalib(knot_sample_size=0, add_knots=[.1,.2,.3,.4,.5])
    sc.fit(xvec, yvec)
    assert(len(sc.knot_vec)==5)

def test_random_and_add_knots():
    npts = 5000
    np.random.seed(42)
    xvec = np.random.uniform(size =npts)
    yvec = np.random.binomial(n=1, p=xvec)
    sc = mli.SplineCalib(knot_sample_size=20, force_knot_endpts=True, add_knots=[.1,.2,.3,.4,.5])
    sc.fit(xvec, yvec)
    t1 = len(sc.knot_vec)==25
    t2 = np.min(xvec) in sc.knot_vec
    t3 = np.max(xvec) in sc.knot_vec
    t4 = .2 in sc.knot_vec
    assert(t1 and t2 and t3)


