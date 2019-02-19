import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def cv_predict_proba(X, y,estimator, cv):
    
    ## Convert from Pandas to Numpy if necessary
    if (type(X)==pd.DataFrame)  or (type(X)==pd.Series):
        X = X.values
    if (type(y)==pd.DataFrame)  or (type(y)==pd.Series):
        y = y.values

    num_classes = len(np.unique(y))
    out_vec=np.zeros((len(y),num_classes))

    #Main loop to do cross-validated predict proba and construct output matrix
    for tr, te in cv.split(X,y):
        estimator.fit(X[tr],y[tr])
        out_vals = estimator.predict_proba(X[te])
        out_vec[te,:] = out_vals
    return out_vec


def cv_score(X, y, estimator, cv, score_fn):
    return(score_fn(y,cv_predict_proba(X,y,estimator,cv)))

def _get_param_settings_from_grid(param_grid):
    num_settings = np.prod([len(i) for i in param_grid.values()])
    pg_tuple = tuple(param_grid.items())
    param_names = [k[0] for k in pg_tuple]
    param_lists = [k[1] for k in pg_tuple]
    param_list_lengths = [len(k) for k in param_lists]
    param_dict_list = []
    for i in range(num_settings):
        indices = _int_to_indices(i,param_list_lengths)
        curr_param_dict = {}
        for k in range(len(param_names)):
            curr_param_dict[param_names[k]]=param_lists[k][indices[k]]
        param_dict_list.append(curr_param_dict)
    return param_dict_list    
    
def _int_to_indices(j,lengths):
    out_list = []
    for i in range(len(lengths)):
        curr_ind = j % lengths[i]
        out_list.append(curr_ind)
        j = j//lengths[i]
    return(out_list)

def grid_search(X,y, model, param_grid, score_fn, verbose=True):
    param_arg_list = _get_param_settings_from_grid(param_grid)
    num_settings = len(param_arg_list)
    param_list_scores = np.zeros(num_settings)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for i in range(num_settings):
        curr_param_dict = param_arg_list[i]
        if verbose:
            print(curr_param_dict)
        model.set_params(**curr_param_dict)
        curr_score=cv_score(X,y,model,skf,score_fn)
        param_list_scores[i]=curr_score
        if verbose:
            print(curr_param_dict,curr_score)

    return(list(zip(param_arg_list,param_list_scores)))

