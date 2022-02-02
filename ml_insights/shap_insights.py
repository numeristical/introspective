import numpy as np
import pandas as pd

try:
    import xgboost as xgb

except ImportError:
    xgb_installed = False

def consolidate_reason_scores(df_ind_expl, dict_map):
    reason_list = dict_map.keys()
    df_rsn = pd.DataFrame(columns = reason_list)
    for reason in reason_list:
        df_rsn[reason] = np.sum(df_ind_expl.loc[:,dict_map[reason]], axis=1)
    return df_rsn 

def get_reason_codes(df_rsn, thresh, direction='greater', delimiter=';'):
    nr, nc = df_rsn.shape
    argsort_mat = np.argsort(-df_rsn.values)
    if (direction=='lesser'):
        num_exceeding_thresh_vec = np.sum(df_rsn.values<=thresh, axis=1)
    else:
        num_exceeding_thresh_vec = np.sum(df_rsn.values>=thresh, axis=1)
    reason_mat = np.array([df_rsn.columns[i] for row in argsort_mat for i in row ]).reshape(nr,nc)
    reason_vec = np.array([delimiter.join(list(reason_mat[j][:num_exceeding_thresh_vec[j]])) for j in range(nr)])
    return reason_vec

def cv_column_shap(xgbcv, X_pr, fn):
    results = np.zeros((X_pr.shape[0], xgbcv.num_features+1))
    fold_set = np.unique(fn)
    for fold in fold_set:
        X_te = xgb.DMatrix(X_pr[fn == fold].values)
        fold_results = xgbcv.model_dict[fold].get_booster().predict(X_te, pred_contribs=True, validate_features=False)
        results[fn==fold] = fold_results
    return results
    
def predict_reasons_cv(xgbcv, X_pr, fn, reason_map, thresh, delimiter=';'):
    shap_val_mat = cv_column_shap(xgbcv, X_pr, fn)
    df_shap_val = pd.DataFrame(shap_val_mat[:,:-1], columns = X_pr.columns)
    df_reason_scores = consolidate_reason_scores(df_shap_val,reason_map)
    reason_list_vec = get_reason_codes(df_reason_scores, thresh, delimiter=delimiter)
    return(reason_list_vec)

def predict_reason_strings(xgbmodel, X_pr, reason_map, thresh, delimiter=';', direction='greater'):
    X_pr_dmat = xgb.DMatrix(X_pr)
    shap_val_mat = xgbmodel.get_booster().predict(X_pr_dmat, pred_contribs=True, validate_features=False)
    df_shap_val = pd.DataFrame(shap_val_mat[:,:-1], columns = X_pr.columns)
    df_reason_scores = consolidate_reason_scores(df_shap_val,reason_map)
    reason_list_vec = get_reason_codes(df_reason_scores, thresh, direction=direction, delimiter=delimiter)
    return(reason_list_vec)

def get_reason_score_matrix(xgbmodel, X_pr, validate=False):
    if (type(X_pr)==pd.DataFrame):
        X_test_dmat = xgb.DMatrix(X_pr)
        reason_list = list(X_pr.columns) + ['Intercept']
        reas_mat = xgbmodel.get_booster().predict(X_test_dmat, pred_contribs=True, validate_features=validate)
    else:
        reason_list = ['f'+str(i) for i in range(X_pr.shape[1])] + ['Intercept']
        X_test_dmat = xgb.DMatrix(X_pr, feature_names = reason_list[:-1])
        reas_mat = xgbmodel.get_booster().predict(X_test_dmat, pred_contribs=True, validate_features=validate)
    return(pd.DataFrame(reas_mat, columns=reason_list))

# def augment_tree(tree_dict):
#     if 'leaf' in tree_dict.keys():
#         value = tree_dict['leaf']
#         tree_dict['value_at_node'] = value
#         return value
#     else:
#         a0 = tree_dict['children'][0]['cover']
#         a1 = tree_dict['children'][1]['cover']
#         value = (a0 * augment_tree(tree_dict['children'][0]) + a1 * augment_tree(tree_dict['children'][1]))/(a0 + a1)
        
#         tree_dict['value_at_node'] = value
#         return value

