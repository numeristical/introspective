"""
Package Docuemntation
"""
# -*- coding: utf-8 -*-

from .insights import ModelXRay, explain_prediction_difference, explain_prediction_difference_xgboost
from .calibration import SplineCalibratedClassifierCV
from .splinecalib import SplineCalib
from .calibration_utils import prob_calibration_function, prob_calibration_function_multiclass
from .calibration_utils import compact_logit,inverse_compact_logit,plot_prob_calibration,plot_reliability_diagram
from .calibration_utils import get_stratified_foldnums,cv_predictions
from .calibration_utils import my_logit, my_logistic
from .modeling_utils import plot_pr_curve,plot_pr_curves,histogram_pair
from .modeling_utils import ice_plot,get_range_dict
from .shap_insights import consolidate_reason_scores, get_reason_codes, cv_column_shap, predict_reason_strings, predict_reasons_cv, get_reason_score_matrix
from . import metrics
from .CVModel import CVModel

__version__ = '0.1.7'
