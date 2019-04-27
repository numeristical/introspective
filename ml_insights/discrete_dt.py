"""Decision Tree based on Discrete Graph structure"""
import numpy as np
import pandas as pd
from .graphs import *

# try:
#     from sklearn.model_selection import StratifiedKFold
# except:
#     from sklearn.cross_validation import StratifiedKFold


class DiscreteGraphDecisionTree(object):

    def __init__(self, feature_graphs, features='auto', min_size_split=300, max_depth=3):
        self.feature_graphs = feature_graphs
        self.min_size_split=min_size_split
        self.max_depth=max_depth
        if features=='auto':
            self.features=list(self.feature_graphs.keys())

    def fit(self, X_train, y_train):
        self.dec_tree={}
        self.dec_tree['root']=_process_tree_node(X_train, y_train, self.feature_graphs, self.min_size_split, 0, self.max_depth)

    def predict(self, X_test):
            out_vec = np.zeros(X_test.shape[0])
            for i in range(X_test.shape[0]):
                pointer = self.dec_tree['root']
                while 'split_feature' in list(pointer['data_for_node'].keys()):
                    if X_test.iloc[i][pointer['data_for_node']['split_feature']] in pointer['data_for_node']['left_split']:
                        pointer = pointer['left_node']
                    elif X_test.iloc[i][pointer['data_for_node']['split_feature']] in pointer['data_for_node']['right_split']:
                        pointer = pointer['right_node']
                    else:
                        print("value in neither split: using interior value at row {}".format(i))
                        break
                out_vec[i] = pointer['data_for_node']['node_mean_value']
            return(out_vec)
   
def _process_tree_node(X_train_node, y_train_node, feature_graphs, min_size_split, curr_depth, max_depth, passed_value=None):
    node_dict = {}
    node_dict['data_for_node']={}
    node_dict['data_for_node']['num_data_points'] = X_train_node.shape[0]
    if passed_value is None:
        node_dict['data_for_node']['node_mean_value'] = np.mean(y_train_node)
    else:
        node_dict['data_for_node']['node_mean_value'] = passed_value
    node_dict['data_for_node']['depth'] = curr_depth
    num_distinct_values = {}
    for feature,graph in feature_graphs.items():
        num_distinct_values[feature] = len(np.unique(X_train_node[feature]))

    ## Remove features from consideration if they only have <=1 distinct values in the current data
    features_to_search = [feature for feature in X_train_node.columns if num_distinct_values[feature]>1]
    if features_to_search==[]:
        return(node_dict)
    best_loss_score = np.inf
    best_left_split = None    
    best_right_split = None    
    best_split_feature = None
    for feature in features_to_search:
        feature_graph = feature_graphs[feature]
        if not is_connected(feature_graph):
            print("Warning: induced graph not connected")
        possible_splits = feature_graph.enumerate_all_partitions()
        for left_split in possible_splits:
            right_split = frozenset(feature_graph.vertices - left_split)
            mask_left = X_train_node[feature].isin(left_split)
            mask_right = X_train_node[feature].isin(right_split)
            left_data = X_train_node[mask_left]
            right_data = X_train_node[mask_right]
            left_val, right_val, loss_score = _score_data_split(mask_left, mask_right, y_train_node)
            if (loss_score < best_loss_score):
                best_loss_score = loss_score
                best_split_feature = feature
                best_left_split = left_split
                best_right_split = right_split
                best_left_val = left_val
                best_right_val = right_val
    node_dict['data_for_node']['left_split'] = best_left_split
    node_dict['data_for_node']['right_split'] = best_right_split
    node_dict['data_for_node']['loss_score'] = best_loss_score
    node_dict['data_for_node']['split_feature'] = best_split_feature       
    curr_split_feature = best_split_feature
    left_split_train_mask = X_train_node[curr_split_feature].isin(best_left_split)
    right_split_train_mask = X_train_node[curr_split_feature].isin(best_right_split)
    left_X_data = X_train_node[left_split_train_mask]
    left_y_data = y_train_node[left_split_train_mask]
    right_X_data = X_train_node[right_split_train_mask]
    right_y_data = y_train_node[right_split_train_mask]
    feature_graphs_left = feature_graphs.copy()
    feature_graphs_left[curr_split_feature] = get_induced_subgraph(feature_graphs_left[curr_split_feature], best_left_split)
    feature_graphs_right = feature_graphs.copy()
    feature_graphs_right[curr_split_feature] = get_induced_subgraph(feature_graphs_right[curr_split_feature], best_right_split)
    if (left_X_data.shape[0]>min_size_split) and (curr_depth<max_depth-1):
        node_dict['left_node'] = _process_tree_node(left_X_data,left_y_data,feature_graphs_left, min_size_split, curr_depth+1, max_depth, passed_value=best_left_val)
    else:
        node_dict['left_node'] = {}
        node_dict['left_node']['data_for_node'] = {}
        node_dict['left_node']['data_for_node']['node_mean_value'] = best_left_val
        node_dict['left_node']['data_for_node']['num_data_points'] = left_X_data.shape[0]
        node_dict['left_node']['data_for_node']['depth'] = curr_depth+1
    if (right_X_data.shape[0]>min_size_split) and (curr_depth<max_depth-1):
        node_dict['right_node'] = _process_tree_node(right_X_data,right_y_data,feature_graphs_right, min_size_split, curr_depth+1, max_depth, passed_value=best_right_val)
    else:
        node_dict['right_node'] = {}
        node_dict['right_node']['data_for_node'] = {}
        node_dict['right_node']['data_for_node']['node_mean_value'] = best_right_val
        node_dict['right_node']['data_for_node']['num_data_points'] = right_X_data.shape[0]
        node_dict['right_node']['data_for_node']['depth'] = curr_depth+1
    return(node_dict)

def root_mean_squared_error(vec1, vec2):
    return np.sqrt(np.mean((vec1-vec2)**2))

# def _score_data_split(mask_left, mask_right, outcome_vec, loss_fn = root_mean_squared_error):
#     value_left = np.mean(outcome_vec[mask_left])
#     value_right = np.mean(outcome_vec[mask_right])
#     t_vec = np.zeros(len(outcome_vec))
#     t_vec[mask_left] = value_left
#     t_vec[mask_right] = value_right
#     loss_score = loss_fn(outcome_vec, t_vec)
#     return value_left, value_right, loss_score

def _score_data_split(mask_left, mask_right, outcome_vec, loss_fn = root_mean_squared_error):
    prob_equal = .1
    m1 = np.sum(outcome_vec[mask_left])
    n1 = np.sum(mask_left)
    m2 = np.sum(outcome_vec[mask_right])
    n2 = np.sum(mask_right)
    num1 = m1*np.log(((m1/n1)/((m1+m2)/(n1+n2)))) + (n1-m1) * np.log((((n1-m1)/n1)/((((n1+n2)-(m1+m2)))/(n1+n2))))
    num2 = m2*np.log(((m2/n2)/((m1+m2)/(n1+n2)))) + (n2-m2) * np.log((((n2-m2)/n2)/((((n1+n2)-(m1+m2)))/(n1+n2))))
    log_odds = num1+num2 +np.log(1-prob_equal) + np.log(prob_equal)
    p_same_post = 1-(np.exp(log_odds)/(1+np.exp(log_odds)))
    value_left = p_same_post*((m1+m2)/(n1+n2)) + (1-p_same_post)*((m1)/(n1))
    value_right = p_same_post*((m1+m2)/(n1+n2)) + (1-p_same_post)*((m2)/(n2))
    t_vec = np.zeros(len(outcome_vec))
    t_vec[mask_left] = value_left
    t_vec[mask_right] = value_right
    loss_score = -log_odds
    return value_left, value_right, loss_score

