import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import ml_insights as mli
from sklearn.metrics import roc_auc_score, log_loss

data_path = op.join(mli.__path__[0], 'data')


def test_1():
    assert(True)


