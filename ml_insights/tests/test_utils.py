import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import ml_insights as mli
from sklearn.metrics import roc_auc_score, log_loss

data_path = op.join(mli.__path__[0], 'data')


def test_get_range_dict():
    """
    This tests the get_range_dict function on a sample dataframe

    """
    df = pd.read_csv(op.join(data_path,'faux_data.csv'))
    rd = mli.get_range_dict(df, max_pts=172)
    t1 = len(rd['int1']), len(rd['int2']) == (100, 172)
    t2 = len(rd['float1']) == 172
    t3 = len(rd['str1']), len(rd['str2']) == (172, 50)
    assert(t1 and t2 and t3)


