"""
Package Docuemntation
"""
# -*- coding: utf-8 -*-

from .insights import ModelXRay
from .calibration import SplineCalibratedClassifierCV
from .calibration_utils import prob_calibration_function, train_and_calibrate_cv, prob_calibration_function_multiclass
from .calibration_utils import compact_logit,plot_prob_calibration,plot_empirical_probs
from . import metrics 

__version__ = '0.0.16'
