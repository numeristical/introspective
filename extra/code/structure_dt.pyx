# cython: profile=True

"""Decision Tree based on Discrete Graph structure"""
import numpy as np
import pandas as pd
import random
cimport numpy as cnp
from libc.math cimport log as clog
from graphs import *
import copy

class StructureDecisionTree(object):
    """This class represents a tree built on categorical features, each of which contains
    a graph to represent the associated terrain.  Splits will be tried according to the 
    *maximally coarse partitions* returned from the graph class.

    feature_graphs: a dictionary which maps the column names to a graph_undirected object.
                    The graph_undirected must contain vertices for every possible value of that column
                    If the graph contains no edges, it will be treated as one-hot encoded.

    loss_fn: Currently there are three options: 
            'entropy':  will use the information gain to choose the best split (target nust be [0,1])
            'mse': will use (minimum) mean squared error to choose the best split (target must be numeric)
            'gh': This uses the XGBoost method where the first derivative (g) and second derivative (h) of the
                custom loss function must be provided.  In this case, the 'g' values should be passed as y_train
                and the 'h' values passed as y_train_2

    min_size_split: The size, below which, the tree will not consider splitting further.  Default is 2.

    min_leaf_size: The minimum permitted size of a split.  Splits will not be considered if they result
                    in a leaf smaller than min_leaf_size

    max_depth: The maximum depth permitted for the tree.  Setting to 1 means creating 'stumps' (a single split).

    gamma: The minimum improvement required to execute a split (for regularization purposes).  
            If the improvement of a split does not exceed gamma, then the node will not be split.

    reg_lambda: The L1 shrinkage applied to the coefficients, as in XGBoost.

    node_summary_fn: Given a collection of points at the node, what should be the value of the node.  Default is
            to take the mean.

    max_splits_to_search: For a feature, what is the maximum number of splits we should search.  Categorical features
            may have prohibitively many possible splits.  If the number exceeds max_splits_to_search, we randomly choose
            only max_splits_to_search of them to evaluate.  Default is infinity (search all splits)
    """

    def __init__(self, feature_configs, feature_graphs, min_size_split=2, 
                min_leaf_size = 2, max_depth=3, gamma=0,
                reg_lambda=1):
        self.dec_tree={}
        self.feature_configs = feature_configs
        self.dec_tree['feature_graphs'] = feature_graphs
        self.num_leafs = 0
        self.min_size_split=min_size_split
        self.min_leaf_size=min_leaf_size
        self.max_depth=max_depth
        self.gamma=gamma
        self.reg_lambda=reg_lambda
        self.node_summary_fn=_node_summary_gh

    def fit(self, X_train, g_train, h_train):        
        # Tree fitting works through a queue of nodes to process (node_to_proc_list) 
        # The initial node is just the root of the tree
        self.node_to_proc_list = [self.dec_tree]
        
        # Initialize values to what they are at the root of the tree
        self.dec_tree['depth']=0
        self.dec_tree['mask'] = np.ones(len(g_train)).astype(bool)
        self.X_train = X_train
        self.g_train = g_train
        self.h_train = h_train

        # Process nodes until none are left to process
        while self.node_to_proc_list:
            node_to_process = self.node_to_proc_list.pop()
            self._process_tree_node(node_to_process)
    
    def predict(self, X_test):
        cdef int i, n=X_test.shape[0]
        cdef dict data_row_dict, pointer, col_to_int_dict
        cdef frozenset left_set  
        
        col_list = list(X_test.columns)
        data_np = X_test.values
        col_to_int_dict = {col_list[i]:i for i in range(len(col_list))}

        # Initialize the output vector to all zeros
        out_vec = np.zeros(X_test.shape[0])
        
        # This iterates through each data point in test set and follows the tree until it
        # reaches a leaf node
        for i in range(n):
            # Put the relevant values for current test point into a dict for quicker lookup
            data_row_dict = {colname:data_np[i,col_to_int_dict[colname]] for colname in col_list}
            pointer = self.dec_tree
            while pointer['node_type']=='interior':
                curr_element = data_row_dict[pointer['split_feature']]
                if pointer['feature_type']=='categ_graphical':
                    left_set = pointer['left_split']
                    if curr_element in left_set:
                        pointer = pointer['left_child']
                    else:
                        pointer = pointer['right_child']
                    continue
                if pointer['feature_type']=='numerical':
                    if curr_element <= pointer['left_split']:
                        pointer = pointer['left_child']
                    else:
                        pointer = pointer['right_child']
                    continue
            out_vec[i] = pointer['node_summary_val']
        return(out_vec)
   
    def _process_tree_node(self, curr_node):
        # Restrict to relevant data for the node in question
        X_train_node = self.X_train[curr_node['mask']]
        
        # Get the associated y-values (or g,h values)
        # and save information about the current node
        g_train_node = self.g_train[curr_node['mask']].values
        h_train_node = self.h_train[curr_node['mask']].values
        curr_node['node_summary_val'] = _node_summary_gh(g_train_node, h_train_node, self.gamma)
        curr_node['num_data_points'] = len(g_train_node)
        g_sum_node = np.sum(g_train_node)
        h_sum_node = np.sum(h_train_node)

        # If we are guaranteed not to split this node further, then mark it as such and move on
        if (curr_node['num_data_points']<self.min_size_split) or (curr_node['depth']>=self.max_depth):
            self._wrap_up_node(curr_node, g_train_node, h_train_node)
            return None
        
        # Determine which features are still "eligible" to be considered
        features_to_search = self.feature_configs.keys()
        
        # print('features_to_search')
        # print(features_to_search)
        # If no features are eligible (e.g. all x-values are identical in all features)
        # Then we similarly summarize the node and move on
        # if features_to_search==[]:
        #     self._wrap_up_node(curr_node, g_train_node, h_train_node)
        #     return None
        
        # best_split_dict holds all the necessary info about a potential split
        best_split_dict = _initialize_best_split_dict()

        # Main loop over features to find best split
        for feature in features_to_search:
            # print('evaluating feature {}'.format(feature))
            best_split_for_feature = evaluate_feature(self.feature_configs[feature], 
                                                        curr_node['feature_graphs'],
                                                        feature,
                                                        X_train_node[feature].values, 
                                                        g_train_node, h_train_node, 
                                                        self.gamma, self.reg_lambda)
            if best_split_for_feature:
                best_split_for_feature['split_feature'] = feature
                if best_split_for_feature['loss_score']<best_split_dict['loss_score']:
                    best_split_dict = best_split_for_feature

        # Execute the split (if a good-enough split is found) otherwise stop
        if best_split_dict['loss_score'] < np.inf:
            self._execute_split(curr_node, best_split_dict, curr_node['feature_graphs'])
        else:
            self._wrap_up_node(curr_node, g_train_node, h_train_node)

    def _wrap_up_node(self, curr_node, g_train_node, h_train_node):
        # Compute summary stats of node and mark it as a leaf
        curr_node['node_summary_val'] = _node_summary_gh(g_train_node, h_train_node, self.reg_lambda)
        curr_node['num_data_points'] = len(g_train_node)
        curr_node['node_type'] = 'leaf'
        self.num_leafs+=1
        curr_node.pop('mask')

    def _execute_split(self, curr_node, best_split_dict, feature_graphs_node):
        if best_split_dict['feature_type']=='numerical':
            self._execute_split_numerical(curr_node, best_split_dict, feature_graphs_node)
        if best_split_dict['feature_type']=='categ_graphical':
            self._execute_split_graphical(curr_node, best_split_dict, feature_graphs_node)

    def _execute_split_numerical(self, curr_node, best_split_dict, feature_graphs_node):
        left_mask = (self.X_train[best_split_dict['split_feature']]<=best_split_dict['left_split']).values
        curr_node['left_split'] = best_split_dict['left_split']
        curr_node['loss_score'] = best_split_dict['loss_score']
        curr_node['split_feature'] = best_split_dict['split_feature']       
        curr_node['node_type'] = 'interior'
        curr_node['feature_type'] = best_split_dict['feature_type']
        curr_mask = curr_node.pop('mask')      

        # Create feature graphs for children
        feature_graphs_left = feature_graphs_node.copy()
        feature_graphs_right = feature_graphs_node.copy()
        # feature_configs_left = copy.deepcopy(feature_configs_node)
        # feature_configs_right = copy.deepcopy(feature_configs_node)

        self._create_children_nodes(curr_node, feature_graphs_left, feature_graphs_right, curr_mask, left_mask)
       
    def _create_children_nodes(self, curr_node, feature_graphs_left, feature_graphs_right, curr_mask, left_mask):
        # Create left and right children
        curr_node['left_child'] = {}
        curr_node['left_child']['depth'] = curr_node['depth'] + 1
        curr_node['left_child']['mask'] = curr_mask & left_mask
        curr_node['left_child']['feature_graphs'] = feature_graphs_left

        curr_node['right_child'] = {}
        curr_node['right_child']['depth'] = curr_node['depth'] + 1
        curr_node['right_child']['mask'] = curr_mask & np.logical_not(left_mask)
        curr_node['right_child']['feature_graphs'] = feature_graphs_right

        # Add left and right children to queue
        self.node_to_proc_list.append(curr_node['left_child'])
        self.node_to_proc_list.append(curr_node['right_child'])


    def _execute_split_graphical(self, curr_node, best_split_dict, feature_graphs_node):

        left_mask = self.X_train[best_split_dict['split_feature']].isin(best_split_dict['left_split']).values

        # record info about current node
        curr_node['left_split'] = best_split_dict['left_split']
        curr_node['right_split'] = feature_graphs_node[best_split_dict['split_feature']].vertices - best_split_dict['left_split']
        curr_node['loss_score'] = best_split_dict['loss_score']
        curr_node['split_feature'] = best_split_dict['split_feature']       
        curr_node['node_type'] = 'interior'
        curr_node['feature_type'] = best_split_dict['feature_type']
        curr_mask = curr_node.pop('mask')      

        # Create feature graphs for children
        feature_graphs_left = feature_graphs_node.copy()
        feature_graphs_left[curr_node['split_feature']] = get_induced_subgraph(feature_graphs_left[curr_node['split_feature']], 
                                                                                curr_node['left_split'])
        feature_graphs_right = feature_graphs_node.copy()
        feature_graphs_right[curr_node['split_feature']] = get_induced_subgraph(feature_graphs_right[curr_node['split_feature']], 
                                                                                curr_node['right_split'])

        # feature_graphs_left = feature_graphs_node.copy()
        # curr_graph = feature_configs_node[curr_node['split_feature']]['graph']
        # feature_configs_left[curr_node['split_feature']]['graph'] = get_induced_subgraph(curr_graph, 
        #                                                                         curr_node['left_split'])
        # feature_configs_right = copy.deepcopy(feature_configs_node)
        # feature_configs_right[curr_node['split_feature']]['graph'] = get_induced_subgraph(curr_graph, 
        #                                                                         curr_graph.vertices -curr_node['left_split'])

        # print('-------------------------------------------------------')
        # print('Executing Graphical Split')
        # print('Left Split')
        # print(curr_node['left_split'])
        # print('Right Split')
        # print(curr_graph.vertices -curr_node['left_split'])
        # print('-------------------------------------------------------')

        self._create_children_nodes(curr_node, feature_graphs_left, feature_graphs_right, curr_mask, left_mask)
    

def _initialize_best_split_dict():
    out_dict = {}
    out_dict['loss_score'] = np.inf
    out_dict['left_split'] = None    
    out_dict['split_feature'] = None
    return(out_dict)


def root_mean_squared_error(vec1, vec2):
    return np.sqrt(np.mean((vec1-vec2)**2))


def _get_gh_score_num(double g_left,  double g_right, 
                    double h_left, double h_right, double gamma, double reg_lambda):
    return(.5*( ((g_left*g_left)/(h_left+reg_lambda)) + ((g_right*g_right)/(h_right+reg_lambda)) - (((g_left + g_right)*(g_left + g_right))/(h_left + h_right+reg_lambda)))-gamma)


def _get_gh_score_array(cnp.ndarray[double] g_left, cnp.ndarray[double]g_right, 
                    cnp.ndarray[double]h_left, cnp.ndarray[double]h_right, double gamma, double reg_lambda):
    return(.5*( ((g_left*g_left)/(h_left+reg_lambda)) + ((g_right*g_right)/(h_right+reg_lambda)) - (((g_left + g_right)*(g_left + g_right))/(h_left + h_right+reg_lambda)))-gamma)

def _node_summary_gh(y_vec_g, y_vec_h, reg_lambda):
    out_val = -np.sum(y_vec_g)/(np.sum(y_vec_h)+reg_lambda)
    return(out_val)


def evaluate_feature(feature_config, feature_graphs, feature_name,
                feature_vec_node, g_train_node, h_train_node, gamma, reg_lambda):

    feature_type = feature_config['feature_type']
    if feature_type=='numerical':
        return _evaluate_feature_numerical(feature_config, feature_vec_node, 
                                            g_train_node, h_train_node, gamma, reg_lambda)
    if feature_type=='categ_graphical':
        return _evaluate_feature_graphical(feature_config, feature_graphs[feature_name], feature_vec_node,
                                            g_train_node, h_train_node, gamma, reg_lambda)

    # if feature_type=='categ_one_hot':
    #     return _evaluate_feature_one_hot(feature_config, feature_vec_node,
    #                                         g_train_node, h_train_node, gamma, reg_lambda)

def _evaluate_feature_numerical(feature_config, feature_vec, g_vec, h_vec, gamma, reg_lambda):
    splits_to_eval = _get_numerical_splits(feature_vec)
    if len(splits_to_eval)>0:
        split_res = feature_config['split_res'] if 'split_res' in feature_config.keys() else np.Inf
        split_count = len(splits_to_eval)
        if split_res<split_count:
            splits_to_eval = splits_to_eval[np.unique(np.random.randint(split_count, size=split_res))]
        best_loss, best_split_val = _evaluate_numerical_splits(feature_vec, g_vec, h_vec, splits_to_eval, gamma, reg_lambda)
        best_split_of_feat={}
        best_split_of_feat['loss_score'] = best_loss
        best_split_of_feat['left_split'] = best_split_val    
        best_split_of_feat['feature_type'] = 'numerical'
        return(best_split_of_feat)
    else:
        return({})


def _get_numerical_splits(feature_vec, prec_digits=16):
    unique_vals = np.sort(pd.unique(feature_vec))
    if len(unique_vals)>1:
        unique_splits = (unique_vals[1:]+unique_vals[:-1])/2
        return unique_splits
    else:
        return []

def _evaluate_numerical_splits(feature_vec, g_vec, h_vec, split_vec, gamma, reg_lambda):
    ## NOTE : need to incorporate min_leaf_size restriction

    bin_result_vec = np.searchsorted(split_vec, feature_vec, side='left')
    g_sum_bins, h_sum_bins = get_bin_sums_c(g_vec, h_vec, bin_result_vec, len(split_vec)+1)
    g_sum_total, g_sum_left, g_sum_right = get_left_right_sums(g_sum_bins)
    h_sum_total, h_sum_left, h_sum_right = get_left_right_sums(h_sum_bins)
    score_vec = (-1)*_get_gh_score_array(g_sum_left, g_sum_right, h_sum_left, h_sum_right, gamma, reg_lambda)
    # if (len(score_vec)!=len(split_vec)):
    #     print('score_vec has length {}'.format(len(score_vec)))
    #     print('split_vec has length {}'.format(len(split_vec)))

    best_loss, best_split_val = get_best_vals(score_vec, split_vec)
    return best_loss, best_split_val

def get_best_vals(score_vec, split_vec):
    best_loss = np.min(score_vec)
    best_split_index = np.argmin(score_vec)
    best_split_val =  split_vec[np.argmin(score_vec)]
    return best_loss, best_split_val

def get_bin_sums(g_vec, h_vec, bin_result_vec, out_vec_size):
    g_sum_bins = np.zeros(out_vec_size)
    h_sum_bins = np.zeros(out_vec_size)
    for i,bin_ind in enumerate(bin_result_vec):
        g_sum_bins[bin_ind]+=g_vec[i]
        h_sum_bins[bin_ind]+=h_vec[i]
    return g_sum_bins, h_sum_bins

def get_bin_sums_c(cnp.ndarray[double] g_vec, cnp.ndarray[double] h_vec, 
                    cnp.ndarray[long] bin_result_vec, long out_vec_size):
    cdef int i
    cdef int m = bin_result_vec.shape[0]

    cdef cnp.ndarray[double] g_sum_bins = np.zeros(out_vec_size)
    cdef cnp.ndarray[double] h_sum_bins = np.zeros(out_vec_size)
    
    for i in range(m):
        g_sum_bins[bin_result_vec[i]]+=g_vec[i]
        h_sum_bins[bin_result_vec[i]]+=h_vec[i]
    return g_sum_bins, h_sum_bins


def get_left_right_sums(bin_sums):
    sum_total = np.sum(bin_sums)
    sum_left = (np.cumsum(bin_sums))[:-1]
    sum_right = sum_total - sum_left
    return sum_total, sum_left, sum_right

def _evaluate_feature_graphical(feature_config, feature_graph, feature_vec_node, 
                                g_train_node, h_train_node, gamma, reg_lambda):
    # NOTE: need to incorporate min_leaf_size restriction

    msac = feature_config['msac']
    msts = feature_config['split_res']
    # Query the graph structure to get the possible splits
    # print('len(feature_graph.mc_partitions)={}'.format(len(feature_graph.mc_partitions)))
    if (len(feature_graph.mc_partitions)>0):
        possible_splits = feature_graph.return_mc_partitions()
    else:
        # print('vertices = {}'.format(feature_graph.vertices))
        # print('edges = {}'.format(feature_graph.edges))
        possible_splits = feature_graph.return_contracted_partitions(max_size_after_contraction=msac)
    nps = len(possible_splits)
    # print('nps={}'.format(nps))
    if (nps>msts):
        # Randomly choose (with replacement) a subset of possible splits
        index_range = np.random.randint(0,nps,msts)
    else:
        index_range = range(nps)

    
    best_split_of_feat = {}
    best_split_of_feat['loss_score'] = np.Inf
    g_sum = np.sum(g_train_node)
    h_sum = np.sum(h_train_node)
    # Loop within values of each feature
    for index in index_range:
        curr_partition = list(possible_splits[index])
        left_split = curr_partition[0]
        right_split = curr_partition[1]
        mask_left = np.array([x in left_split for x in feature_vec_node])
        curr_loss = _score_split(mask_left, g_train_node, h_train_node, g_sum, h_sum,
                                 gamma, reg_lambda)
        # print('Evaluating split')
        # print(left_split)
        # print('vs')
        # print(right_split)
        # print('loss_score = {}'.format(curr_loss))
        # print('----')

        if curr_loss < best_split_of_feat['loss_score']:
            best_split_of_feat['loss_score'] = curr_loss
            best_split_of_feat['left_split'] = left_split
            best_split_of_feat['feature_type'] = 'categ_graphical'
    return(best_split_of_feat)

def _score_split(mask_left, g_train_node, h_train_node, g_sum, h_sum, gamma, reg_lambda):
    # cdef double loss_score, g_left, g_right, h_left, h_right, vec_len

    vec_len = len(g_train_node)
    g_left = np.sum(g_train_node[mask_left])
    g_right = g_sum - g_left
    h_left = np.sum(h_train_node[mask_left])
    h_right = h_sum - h_left
    loss_score = -1.0 * _get_gh_score_num(g_left, g_right, h_left, h_right, gamma, reg_lambda)
    if loss_score>=0:
        loss_score = np.inf
    return loss_score




