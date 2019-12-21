# cython: profile=True

"""Decision Tree Gradient Boosting based on Discrete Graph structure"""
import numpy as np
import pandas as pd
cimport numpy as cnp
from libc.math cimport log as clog
from discrete_dt import *
from graphs import *
from sklearn.metrics import log_loss, mean_squared_error


class DiscreteGraphGB(object):

    def __init__(self, num_trees, feature_graphs,  mode='classification', loss_fn = 'entropy', min_size_split=2, min_leaf_size = 1, max_depth=3, gamma=0,
                     reg_lambda=1, node_summary_fn = np.mean, learning_rate=.1, max_splits_to_search=np.Inf, msac=100):
        self.num_trees = num_trees
        self.num_trees_for_prediction = num_trees
        self.dec_tree_list = []
        self.feature_graphs = feature_graphs
        self.min_size_split=min_size_split
        self.min_leaf_size=min_leaf_size
        self.max_depth=max_depth
        self.gamma=gamma
        self.node_summary_fn=node_summary_fn
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.max_splits_to_search = max_splits_to_search
        self.msac = msac
        self.mode = mode
        if loss_fn == 'entropy':
            self.loss_fn_der_1 = _entropy_link_der_1
            self.loss_fn_der_2 = _entropy_link_der_2
        if loss_fn == 'mse':
            self.loss_fn_der_1 = _mse_der_1
            self.loss_fn_der_2 = _mse_der_2
        # if features=='auto':
        #     self.features=list(self.dec_tree['feature_graphs'].keys())

    def fit(self, X_train, y_train, eval_set = None, eval_freq=10, 
                early_stop_past_steps=0, choose_best_eval=True):
        # cdef int i, n =self.num_trees
        self.eval_freq=eval_freq
        eval_len = np.floor(self.num_trees/self.eval_freq).astype(int)
        self.eval_results = np.zeros(eval_len)
        n =self.num_trees
        self.initial_pred = np.mean(y_train)
        stop_now=False
        if eval_set is not None:
            X_valid = eval_set[0]
            y_valid = eval_set[1]
        for i in range(n):
            
            # Get predictions of current model
            if i==0:
                curr_answer = self.initial_pred * np.ones(len(y_train))
                if eval_set is not None:
                    curr_test_answer = self.initial_pred * np.ones(len(y_valid))
                    if self.mode == 'classification':
                        curr_loss= log_loss(y_valid, 1/(1+np.exp(-curr_test_answer)))
                        print("i=0, test_set_log_loss = {}".format(curr_loss))
                    else:
                        curr_loss= mean_squared_error(y_valid, curr_test_answer)
                        print("i=0. test_set_mse = {}".format(curr_loss))
                    
            else:
                curr_answer = curr_answer + self.learning_rate * self.dec_tree_list[i-1].predict(X_train) 
                if eval_set is not None:
                    curr_test_answer = curr_test_answer + self.learning_rate * self.dec_tree_list[i-1].predict(X_valid)
                    if ((i+1)%self.eval_freq==1):
                        if self.mode == 'classification':
                            curr_loss= log_loss(y_valid, 1/(1+np.exp(-curr_test_answer)))
                            print("i={}, test_set_log_loss = {}".format(i,curr_loss))
                        else:
                            curr_loss= mean_squared_error(y_valid, curr_test_answer)
                            print("i={}, test_set_mse = {}".format(i,curr_loss))
                        
                        curr_step=np.floor((i+1)/self.eval_freq).astype(int) -1
                        self.eval_results[curr_step]=curr_loss
                        if curr_step>early_stop_past_steps:        
                            compare_loss = np.min(self.eval_results[:curr_step-early_stop_past_steps+1])
                            if (curr_loss>compare_loss):
                                stop_now=True
                                print("Stopping early: curr_loss of {} exceeds compare_loss of {}".format(curr_loss, compare_loss))
            if stop_now:        
                if choose_best_eval:
                    self.num_trees_for_prediction = (np.argmin(self.eval_results[:curr_step+1])+1)*eval_freq
                break

            # Get first and second derivatives
            y_g_vec = self.loss_fn_der_1(y_train, curr_answer)
            y_h_vec = self.loss_fn_der_2(y_train, curr_answer)


           # Sample the data to use for this tree
            
            num_rows = X_train.shape[0]
            rows_to_use = np.random.choice(range(num_rows), num_rows, replace=True)
            if type(X_train)==pd.DataFrame:
                X_train_to_use = X_train.iloc[rows_to_use]
            elif type(X_train)==np.ndarray:
                X_train_to_use = X_train[rows_to_use]
            else:
                print('unknown format for X_train')
            #y_original_train_to_use = y_train.sample(X_train.shape[0], random_state=rs, replace=True)
            if type(y_g_vec)==pd.Series:
                y_g_to_use = y_g_vec.iloc[rows_to_use]
            elif type(y_g_vec)==np.ndarray:
                y_g_to_use = y_g_vec[rows_to_use]
            else:
                print('unknown format for y_g_vec')

            if type(y_h_vec)==pd.Series:
                y_h_to_use = y_h_vec.iloc[rows_to_use]
            elif type(y_h_vec)==np.ndarray:
                y_h_to_use = y_h_vec[rows_to_use]
            else:
                print('unknown format for y_h_vec')

            self.dec_tree_list.append(DiscreteGraphDecisionTree(feature_graphs=self.feature_graphs,loss_fn = 'gh',
                                                 min_size_split = self.min_size_split, min_leaf_size=self.min_leaf_size, 
                                                 gamma=self.gamma, max_depth=self.max_depth, 
                                                 node_summary_fn = self.node_summary_fn, 
                                                 max_splits_to_search = self.max_splits_to_search, msac=self.msac))
            self.dec_tree_list[i].fit(X_train_to_use, y_g_to_use, y_h_to_use)


    def predict(self, X_test, num_trees_to_use=0):
            cdef int i
            if num_trees_to_use==0:
                num_trees_to_use=self.num_trees_for_prediction
            out_vec = self.initial_pred*np.ones(X_test.shape[0])
            for i in range(num_trees_to_use):
                out_vec = out_vec + self.learning_rate * self.dec_tree_list[i].predict(X_test)
            if self.mode=='classification':
                return(1/(1+np.exp(-out_vec)))
            else:
                return(out_vec)

def _entropy_der_1(y_true, y_pred, eps=1e-15):
    y_pred = np.maximum(y_pred, eps)
    y_pred = np.minimum(y_pred, 1-eps)
    return((-(y_true/y_pred) + (1-y_true)/(1-y_pred)))

def _entropy_der_2(y_true, y_pred, eps=1e-15):
    y_pred = np.maximum(y_pred, eps)
    y_pred = np.minimum(y_pred, 1-eps)
    out_vec = (y_true)/(y_pred**2) + ((1-y_true)/((1-y_pred)**2))
    return(out_vec)

def _mse_der_1(y_true, y_pred, eps=1e-15):
    return(2*(y_pred-y_true))

def _mse_der_2(y_true, y_pred, eps=1e-15):
    return(pd.Series(2*np.ones(len(y_pred))))

def _entropy_link_der_1(y_true, z_pred, eps=1e-15):
    return(-y_true*(1/(1+np.exp(z_pred))) + (1-y_true) * (1/(1+np.exp(-z_pred))) )

def _entropy_link_der_2(y_true, z_pred, eps=1e-15):
    return(y_true*(np.exp(z_pred)/((1+np.exp(z_pred))**2)) + (1-y_true) * (np.exp(-z_pred)/((1+np.exp(-z_pred))**2)) )

