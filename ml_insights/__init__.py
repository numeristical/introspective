"""
Package Docuemntation
"""
# -*- coding: utf-8 -*-

from .insights import ModelXRay, explain_prediction_difference, explain_prediction_difference_xgboost
from splinecalib import SplineCalib
from .modeling_utils import get_stratified_foldnums,cv_predictions
from .modeling_utils import plot_pr_curve,plot_pr_curves,histogram_pair
from .modeling_utils import ice_plot,get_range_dict, plot_reliability_diagram
from .shap_insights import consolidate_reason_scores, get_reason_codes, cv_column_shap, predict_reason_strings, predict_reasons_cv, get_reason_score_matrix
from .CVModel import CVModel

__version__ = '1.0.2'
