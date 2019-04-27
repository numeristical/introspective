"""Cross-validated training and prediction."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone

class CVModel(BaseEstimator, ClassifierMixin):

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X_train, y_train, fold_num, train_overall=True, **kwargs):
        """Fits a cross-validated model - a model for each left-out fold plus one overall model.

        X_train: the training predictors
        y_train: the training outcome
        fold_num: the indicator of which fold each row belongs to"""
        self.model_dict = {}
        self.fold_set = np.unique(fold_num)
        self.num_unique_y_values = len(np.unique(y_train))
        self.num_features = X_train.shape[1]

        ## If a DataFrame is given (rather than just an array) then make a note of the column names.  
        ## This way we can match up column names when we use predict_proba.
        if type(X_train) == pd.DataFrame:
            self.fit_columns = np.array(X_train.columns)
        else:
            self.fit_columns = None

        ## Make copies of the estimator for each of the fold models
        for fold in self.fold_set:
            self.model_dict[fold] = clone(self.base_estimator)

        ## Train the separate models, each one leaving out a particular fold in training
        for fold in self.fold_set:
            print("Leave out fold {} and train on the rest".format(fold))
            X_tr = X_train[fold_num != fold]
            y_tr = y_train[fold_num != fold]
            self.model_dict[fold].fit(X_tr, y_tr, **kwargs)

        ## Train the overall model on all of the data
        if train_overall:
            print("Train the overall model".format(fold))
            self.model_dict['overall_model'] = clone(self.base_estimator)
            self.model_dict['overall_model'].fit(X_train, y_train, **kwargs)
        return self

    def predict_proba(self, X_test, fold_num=None, **kwargs):
        """Predict probabilities in cross-validated fashion.

        X_test: the data to predict on
        fold_num: the indicator of which fold a row belongs to / which model variant to use.
        If fold_num is not specified, it will default to use the overall_model
        """
        ## If we have column names and X_test is a DataFrame, then subset X_test to those columns
        ## in the correct order, and error if those columns are not present
        if self.fit_columns is not None:
            if type(X_test) == pd.DataFrame:
                X_test = X_test.loc[:, self.fit_columns]
        
        if fold_num is None:
            #print("no folds specified, using overall_model")
            if 'overall_model' not in self.model_dict.keys():
                #print("Error: overall_model not trained and fold_num not specified")
                return None
            else:
                results = self.model_dict['overall_model'].predict_proba(X_test, **kwargs)
                return results
        else:
            results = np.zeros((X_test.shape[0], self.num_unique_y_values))
            fold_set = np.unique(fold_num)
            for fold in fold_set:
                X_te = X_test[fold_num == fold]
                fold_results = self.model_dict[fold].predict_proba(X_te, **kwargs)
                results[fold_num==fold] = fold_results
            return results

    def predict(self, X_test, fold_num=None, **kwargs):
        """Predict final values in cross-validated fashion.

        X_test: the data to predict on
        fold_num: the indicator of which fold a row belongs to / which model variant to use.
        If fold_num is not specified, it will default to use the overall_model
        
        """

        ## If we have column names and X_test is a DataFrame, then subset X_test to those columns
        ## in the correct order, and error if those columns are not present
        if self.fit_columns is not None:
            if type(X_test) == pd.DataFrame:
                X_test = X_test.loc[:, self.fit_columns]
        
        if fold_num is None:
            #print("no folds specified, using overall_model")
            if 'overall_model' not in self.model_dict.keys():
                print("Error: overall_model not trained and fold_num not specified")
                return None
            else:
                results = self.model_dict['overall_model']
                return results
        else:
            results = np.zeros(X_test.shape[0])
            fold_set = np.unique(fold_num)
            for fold in fold_set:
                X_te = X_test[fold_num == fold]
                fold_results = self.model_dict[fold].predict(X_te, **kwargs)
                results[fold_num==fold] = fold_results
            return results

    def grid_search(self, X, y, fold_ind, param_grid, score_fn, verbose=True):
        param_arg_list = _get_param_settings_from_grid(param_grid)
        num_settings = len(param_arg_list)
        print("Size of grid to search = {} different settings".format(num_settings))
        param_list_scores = np.zeros(num_settings)
        old_self = clone(self.base_estimator)
        for i in range(num_settings):
            print("Fitting setting {} of {}".format(i+1,num_settings))
            curr_param_dict = param_arg_list[i]
            if verbose:
                print(curr_param_dict)
            self.base_estimator.set_params(**curr_param_dict)
            self.fit(X, y, fold_ind, train_overall=False)
            curr_preds = self.predict_proba(X, fold_ind)
            if type(score_fn) == list:
                for j, fn in enumerate(score_fn):
                    curr_score= fn(y, curr_preds)
                    param_arg_list[i]['score_'+str(j)] = curr_score
                    if verbose:
                        print(curr_param_dict,'score function '+str(j)+':',curr_score)
            else:
                curr_score= score_fn(y, curr_preds)
                param_arg_list[i]['score'] = curr_score
                if verbose:
                    print(curr_param_dict,'score function '+':',curr_score)
            param_list_scores[i]=curr_score
        self.base_estimator = old_self
        return param_arg_list


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



