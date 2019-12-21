# cython: profile=True

"""Decision Tree based on Discrete Graph structure"""
import numpy as np
import pandas as pd
import random
cimport numpy as cnp
from libc.math cimport log as clog
from graphs import *


class DiscreteGraphDecisionTree(object):
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

    def __init__(self, feature_graphs, loss_fn = 'entropy', min_size_split=2, min_leaf_size = 2, max_depth=3, gamma=0,
                reg_lambda=1, node_summary_fn = np.mean, max_splits_to_search = np.Inf, msac=13):
        self.dec_tree={}
        self.dec_tree['feature_graphs'] = feature_graphs
        self.num_leafs = 0
        self.min_size_split=min_size_split
        self.min_leaf_size=min_leaf_size
        self.max_depth=max_depth
        self.gamma=gamma
        self.node_summary_fn=node_summary_fn
        self.reg_lambda=reg_lambda
        self.max_splits_to_search = max_splits_to_search
        self.msac = msac
        if loss_fn == 'gh':
            self.loss_fn='gh'
            self.node_summary_fn=_node_summary_gh
            self.split_scorer = _score_data_split_gh
        if loss_fn == 'entropy':
            self.loss_fn='entropy'
            self.split_scorer = _score_data_split_entropy
        if loss_fn == 'mse':
            self.loss_fn='mse'
            self.split_scorer = _score_data_split_mse

    def fit(self, X_train, y_train, y_train_2=None):        
        # Tree fitting works through a queue of nodes to process (node_to_proc_list) 
        # The initial node is just the root of the tree
        self.node_to_proc_list = [self.dec_tree]
        
        # Initialize values to what they are at the root of the tree
        self.dec_tree['depth']=0
        self.dec_tree['mask'] = np.ones(len(y_train))
        self.X_train = X_train
        self.y_train = y_train

        # Special handling for 'gh' loss function
        if self.loss_fn ==  'gh':
            self.y_train_2 = y_train_2
            self.node_summary_fn = _node_summary_gh

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
            # Put the relevant values for current test point into a dict for quick lookup
            data_row_dict = {colname:data_np[i,col_to_int_dict[colname]] for colname in col_list}
            pointer = self.dec_tree
            while pointer['node_type']=='interior':
                curr_element = data_row_dict[pointer['split_feature']]
                left_set = pointer['left_split']
                if curr_element in left_set:
                    pointer = pointer['left_child']
                else:
                    pointer = pointer['right_child']
            out_vec[i] = pointer['node_summary_val']
        return(out_vec)
   
    def _process_tree_node(self, curr_node):
        # Restrict to relevant data for the node in question
        X_train_node = self.X_train[curr_node['mask']>0]
        
        # Get the associated y-values (or g,h values)
        # and save information about the current node
        if self.loss_fn != 'gh':
            y_train_node = self.y_train[curr_node['mask']>0]
            curr_node['node_summary_val'] = self.node_summary_fn(y_train_node)
            curr_node['num_data_points'] = len(y_train_node)
        else:
            y_train_g = self.y_train[curr_node['mask']>0]
            y_train_h = self.y_train_2[curr_node['mask']>0]
            curr_node['node_summary_val'] = _node_summary_gh(y_train_g, y_train_h, self.gamma)
            curr_node['num_data_points'] = len(y_train_g)
            g_sum = np.sum(y_train_g)
            h_sum = np.sum(y_train_h)

        # If we are guaranteed not to split this node further, then mark it as such and move on
        if (curr_node['num_data_points']<self.min_size_split) or (curr_node['depth']>=self.max_depth):
            if self.loss_fn != 'gh':
                self._wrap_up_node(curr_node, y_train_node)
            else:
                self._wrap_up_node(curr_node, y_train_g, y_train_h)
            return None
        
        # Determine which features are still "eligible" to be considered
        features_to_search = _get_features_to_search(X_train_node, curr_node['feature_graphs'])
        
        # If no features are eligible (e.g. all x-values are identical in all features)
        # Then we similarly summarize the node and move on
        if features_to_search==[]:
            if self.loss_fn != 'gh':
                self._wrap_up_node(curr_node, y_train_node)
            else:
                self._wrap_up_node(curr_node, y_train_g, y_train_h)
            return None
        
        # best_split_dict holds all the necessary info about a potential split
        best_split_dict = _initialize_best_split_dict()

        # Main loop over features to find best split
        for feature in features_to_search:
            feature_graph = curr_node['feature_graphs'][feature]
            if len(feature_graph.edges)==0:     # This means to treat the feature as one-hot encoded
                possible_splits = []
                vert_list = list(feature_graph.vertices)
                # Make a list of splits that are one feature vs the rest (as in one-hot-encoding)
                for i in range(len(vert_list)):
                    tfset = frozenset(vert_list[i:i+1])
                    possible_splits.append(frozenset([tfset,frozenset(vert_list)-tfset]))
                #print(possible_splits)
                index_range = range(len(possible_splits))
            else:
                # Query the graph structure to get the possible splits
                if (len(feature_graph.mc_partitions)>0):
                    possible_splits = feature_graph.return_mc_partitions()
                else:
                    possible_splits = feature_graph.return_contracted_partitions(max_size_after_contraction=self.msac)
                    #print('# possible splits = {}'.format(len(possible_splits)))
                #possible_splits = feature_graph.return_mc_partitions()
                if (len(possible_splits)>self.max_splits_to_search):
                    # Randomly choose (with replacement) a subset of possible splits
                    index_range = np.random.randint(0,len(possible_splits),self.max_splits_to_search)
                    #print('index_range_len={} msts={}'.format(len(index_range),self.max_splits_to_search))
                else:
                    index_range = range(len(possible_splits))

            curr_feature_vec = X_train_node[feature].values
            
            # Loop within values of each feature
            for index in index_range:
                curr_partition = list(possible_splits[index])
                left_split = curr_partition[0]
                if self.loss_fn != 'gh':
                    curr_split_dict = _eval_curr_split_dict(curr_feature_vec, y_train_node, curr_node['feature_graphs'], 
                                                            feature, left_split, self.split_scorer, self.min_leaf_size, self.gamma)
                else:
                    curr_split_dict = _eval_curr_split_dict(curr_feature_vec, y_train_g, curr_node['feature_graphs'], 
                                                            feature, left_split, self.split_scorer, self.min_leaf_size, self.gamma, 
                                                            y_train_2_node = y_train_h, is_gh=True, g_sum=g_sum, h_sum=h_sum)

                best_split_dict = _compare_curr_to_best(curr_split_dict, best_split_dict)


        if best_split_dict['best_loss_score'] < np.inf:
            # Execute the split
            left_mask = self.X_train[best_split_dict['best_split_feature']].isin(best_split_dict['best_left_split']).values
            right_mask = self.X_train[best_split_dict['best_split_feature']].isin(best_split_dict['best_right_split']).values
            self.perform_split_on_node(curr_node, best_split_dict, curr_node['feature_graphs'], left_mask, right_mask)
        else:
            if self.loss_fn != 'gh':
                self._wrap_up_node(curr_node, y_train_node)
            else:
                self._wrap_up_node(curr_node, y_train_g, y_train_h)

        return None

    def _wrap_up_node(self, curr_node, y_train_node, y_train_2_node=None):
        # Compute summary stats of node and mark it as a leaf
        if self.loss_fn!='gh':
            curr_node['node_summary_val'] = self.node_summary_fn(y_train_node)
        else:
            curr_node['node_summary_val'] = _node_summary_gh(y_train_node, y_train_2_node, self.reg_lambda)
        curr_node['num_data_points'] = len(y_train_node)
        curr_node['node_type'] = 'leaf'
        self.num_leafs+=1
        curr_node.pop('mask')

    def perform_split_on_node(self, curr_node, best_split_dict, feature_graphs_node, left_mask, right_mask):
        # record info about current node
        curr_node['left_split'] = best_split_dict['best_left_split']
        curr_node['right_split'] = best_split_dict['best_right_split']
        curr_node['loss_score'] = best_split_dict['best_loss_score']
        curr_node['split_feature'] = best_split_dict['best_split_feature']       
        curr_node['node_type'] = 'interior'
        curr_mask = curr_node.pop('mask')      

        # Create feature graphs for children
        feature_graphs_left = feature_graphs_node.copy()
        feature_graphs_left[curr_node['split_feature']] = get_induced_subgraph(feature_graphs_left[curr_node['split_feature']], 
                                                                                curr_node['left_split'])
        feature_graphs_right = feature_graphs_node.copy()
        feature_graphs_right[curr_node['split_feature']] = get_induced_subgraph(feature_graphs_right[curr_node['split_feature']], 
                                                                                curr_node['right_split'])
        # Create left and right children
        curr_node['left_child'] = {}
        curr_node['left_child']['depth'] = curr_node['depth'] + 1
        curr_node['left_child']['mask'] = curr_mask * left_mask
        curr_node['left_child']['feature_graphs'] = feature_graphs_left

        curr_node['right_child'] = {}
        curr_node['right_child']['depth'] = curr_node['depth'] + 1
        curr_node['right_child']['mask'] = curr_mask * right_mask
        curr_node['right_child']['feature_graphs'] = feature_graphs_right

        # Add left and right children to queue
        self.node_to_proc_list.append(curr_node['left_child'])
        self.node_to_proc_list.append(curr_node['right_child'])

       
def _get_features_to_search(X_train_node, feature_graphs_node):
    num_distinct_values = {}
    for feature,graph in feature_graphs_node.items():
        num_distinct_values[feature] = len(np.unique(X_train_node[feature]))

    ## Remove features from consideration if they only have <=1 distinct values in the current data
    features_to_search = [feature for feature in X_train_node.columns if num_distinct_values[feature]>1]
    return(features_to_search)


def _initialize_best_split_dict():
    out_dict = {}
    out_dict['best_loss_score'] = np.inf
    out_dict['best_left_split'] = None    
    out_dict['best_right_split'] = None    
    out_dict['best_split_feature'] = None
    return(out_dict)

def _eval_curr_split_dict(curr_feature_vec, y_train_node, feature_graphs_node, feature, frozenset left_split, split_scorer, min_leaf_size, gamma, 
                            y_train_2_node=None, is_gh=False, g_sum=0, h_sum=0):
    cdef dict out_dict
    cdef frozenset temp_set
    cdef list temp_list

    out_dict = {}
    out_dict['left_split'] = left_split
    out_dict['feature'] = feature
    out_dict['right_split'] = frozenset(feature_graphs_node[feature].vertices - left_split)
    temp_set = out_dict['left_split']
    temp_list = [x in temp_set for x in curr_feature_vec]
    out_dict['mask_left'] = np.array(temp_list)    
    out_dict['mask_right'] = np.logical_not(out_dict['mask_left'])
    if is_gh==False:
        out_dict['loss_score'] = split_scorer(out_dict['mask_left'], out_dict['mask_right'], y_train_node.values, min_leaf_size, gamma)
    else:
        out_dict['loss_score'] = split_scorer(out_dict['mask_left'], out_dict['mask_right'], y_train_node.values, y_train_2_node.values,
                                                                     min_leaf_size, gamma, g_sum, h_sum)

    return(out_dict)

def _compare_curr_to_best(curr_split_dict, best_split_dict):
    if (curr_split_dict['loss_score'] < best_split_dict['best_loss_score']):
        best_split_dict['best_loss_score'] = curr_split_dict['loss_score']
        best_split_dict['best_split_feature'] = curr_split_dict['feature']
        best_split_dict['best_left_split'] = curr_split_dict['left_split']
        best_split_dict['best_right_split'] = curr_split_dict['right_split']
    return(best_split_dict)

def root_mean_squared_error(vec1, vec2):
    return np.sqrt(np.mean((vec1-vec2)**2))


def _score_data_split_mse(mask_left, mask_right, outcome_vec, min_leaf_size, gamma, eps=.0001):

    cdef double mean_left, mean_right,mean_overall,loss_score, n1, n2

    n1 = np.sum(mask_left)
    n2 = np.sum(mask_right)
    if np.minimum(n1, n2)<min_leaf_size:
        return np.inf
    mean_left = np.mean(outcome_vec[mask_left])
    mean_right = np.mean(outcome_vec[mask_right])
    mean_overall = np.mean(outcome_vec)
    t_vec = np.zeros(len(outcome_vec))
    t_vec[mask_left] = mean_left
    t_vec[mask_right] = mean_right
    agg_vec = np.mean(outcome_vec) * np.ones(len(outcome_vec))
    loss_score = root_mean_squared_error(outcome_vec, t_vec) - root_mean_squared_error(outcome_vec, agg_vec)
    loss_score = loss_score-gamma
    if loss_score>=0:
        loss_score = np.inf
    return loss_score



def _score_data_split_entropy(mask_left, mask_right, outcome_vec, min_leaf_size, gamma, eps=.0001):

    cdef double m1,n1,m2,n2,num1,num1a,num2,num2a,lik_rat,loss_score

    m1 = np.sum(outcome_vec[mask_left])+eps
    n1 = np.sum(mask_left)+eps
    m2 = np.sum(outcome_vec[mask_right])+eps
    n2 = np.sum(mask_right)+eps
    if np.minimum(n1, n2)<min_leaf_size:
        return np.inf
    loss_score = -1 * get_lik_rat(m1,n1,m2,n2,eps)
    loss_score - loss_score-gamma
    if loss_score>=0:
        loss_score = np.inf
    return loss_score

cdef double get_lik_rat(double m1, double n1, double m2, double n2, eps):
    cdef double num1, num2
    num1 = m1*clog(((m1/n1)/((m1+m2)/(n1+n2)))+eps) + (n1-m1+eps) * clog((((n1-m1)/n1)/((((n1+n2)-(m1+m2)))/(n1+n2))))
    num2 = m2*clog(((m2/n2)/((m1+m2)/(n1+n2)))) + (n2-m2+eps) * clog((((n2-m2)/n2)/((((n1+n2)-(m1+m2)))/(n1+n2))))
    return num1+num2

def _score_data_split_gh(mask_left, mask_right, outcome_vec_g, outcome_vec_h, min_leaf_size, gamma, g_sum, h_sum):
    cdef double loss_score, g_left, g_right, h_left, h_right, n_left, n_right, vec_len

    vec_len = len(outcome_vec_g)
    g_left = np.sum(outcome_vec_g[mask_left])
    g_right = g_sum - g_left
    #g_right = np.sum(outcome_vec_g[mask_right])
    h_left = np.sum(outcome_vec_h[mask_left])
    h_right = h_sum - h_left
    #h_right = np.sum(outcome_vec_h[mask_right])
    n_left = np.sum(mask_left)
    n_right = vec_len - n_left
    #n_right = np.sum(mask_right)
    if np.minimum(n_left, n_right)<min_leaf_size:
        return np.inf
    loss_score = -1.0 * _get_gh_score(g_left, g_right, h_left, h_right, gamma)
    if loss_score>=0:
        loss_score = np.inf
    return loss_score

cdef double _get_gh_score(double g_left, double g_right, double h_left, double h_right, double gamma):
    return(.5*( ((g_left**2)/(h_left+gamma)) + ((g_right**2)/(h_right+gamma)) - (((g_left + g_right)**2)/(h_left + h_right+gamma)))-gamma)

def _node_summary_gh(y_vec_g, y_vec_h, reg_lambda):
    out_val = -np.sum(y_vec_g)/(np.sum(y_vec_h)+reg_lambda)
    return(out_val)
