"""
Package Docuemntation
"""
# -*- coding: utf-8 -*-

from .insights import ModelXRay, explain_prediction_difference, explain_prediction_difference_xgboost
from .calibration import SplineCalibratedClassifierCV
from .calibration_utils import prob_calibration_function, prob_calibration_function_multiclass
from .calibration_utils import compact_logit,inverse_compact_logit,plot_prob_calibration,plot_reliability_diagram
from . import metrics
from .CVModel import CVModel

__version__ = '0.0.18'
