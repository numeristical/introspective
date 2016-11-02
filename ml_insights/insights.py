import math
import numpy as np
import pandas as pd
from .utils import _gca, is_classifier, is_regressor

def gen_model_pred(model, row, col_idx, values):
    rows = []
    for val in values:
        sim_row = row.copy()
        sim_row[col_idx] = val
        rows.append(sim_row)
    if is_classifier(model):
        y_pred = model.predict_proba(rows)[:,1]
    else:
        y_pred = model.predict(rows)
    return y_pred


def model_xray(model, data, columns=None, resolution=100, normalize_loc=None, **kwargs):
    '''This function executes a model over a broad range of conditions to analyze aspects of its performance.

    For each point in the data set, and for every feature involved of the prediction of the model, a new set of data
    points is created where the chosen feature is varied across its (empirical) range.  These modified data points are
    fed into the model to get a set of model predictions for each feature-data point combination.

    It is desirable that the "data" object passed in be relatively large in size, since the algorithm will make
    some heuristic choices based on the ranges of values it sees.  We suggest using at least 100 data points and preferably
    more than 500.

    It returns a results object, which can then be passed to functions such as feature_effect_summary and
    feature_dependence_plots to gain insight on the how the various features affect the target.  The results
    object can also be used directly by a user who wants to operate at a low-level.

    Parameters
    ----------

    model : A model object from sklearn or similar styled objects.  The `predict` method will be used if it is
        a regression model, while `predict_proba` will be used if it is a (binary) classification model.  Multi-class
        classifiers are not supported at this time.

    data : A DataFrame possessing the same structure that the model would take as an argument.  These methods are designed
        to be used on "test" data (i.e. data that was not used in the training of the model).  However, there is nothing
        structural to prevent it from being used on training data, and there may be some insight gained by doing so.

    columns : a specific subset of columns to be used.  Default is None, which means to use all available columns in *data*

    resolution : how many different "grid points" to use for each feature.  The algorithm will use only the unique values
    it sees in *data* if there are fewer than *resolution* unique values.  Otherwise it will use *resolution* linearly spaced 
    values ranging from the min observed value to the max observed value.
    
    Returns
    -------

    results : The "results" object is a dictionary where the keys are the feature names and the values are a 2-tuple.  This
        object is intended primarily to be passed to other functions to interact with and display the data.  However, advanced
        users may wish to understand and/or use the object directly.

        The first element in the tuple is the set of different feature values that were substituted in for each data point.  The
        second element in the tuple is matrix where the number of rows is the number of data points and the number of columns
        is the number of different feature values.  The (i,j)th element of the matrix is the result of the model prediction when
        data point i has the feature in question set to jth value.
    '''
    ## Convert Pandas DataFrame to nparray explicitly to make life easier
    #print('hello!!!')


    ## Determine the range of values to plot for the chosen column
    if columns is None:
        if type(data) == pd.DataFrame:
            columns = data.columns
        else:
            columns = range(len(data[0]))  # Assuming a 2-D Dataset
    else:
        # Verify that columns is an iterable
        try:
            iterator = iter(columns)
        except TypeError:
            # not iterable
            columns = [columns]
        else:
            # iterable
            pass

    # Build Column Index
    column_nums = []
    if type(data) == pd.DataFrame:
        for column in columns:
            try:
                column_nums.append(data.columns.get_loc(column))
            except KeyError:
                ## TODO
                pass
    else:
        # Column Index and Column Names are the same
        if type(columns[0]) == int:
            column_nums = columns
        else:
            column_nums = range(len(columns))

    if type(data)==pd.DataFrame:
        data = data.values

    results = {}
    num_pts = len(data)
    for column_num, column_name in zip(column_nums, columns):
        if (len(np.unique(data[:,column_num]))> resolution):
            col_values = np.linspace(np.min(data[:,column_num]),np.max(data[:,column_num]),resolution)
        else:
            col_values = np.sort(np.unique(data[:,column_num]))
        ## Define the empty data structure to output
        out_matrix = np.zeros([num_pts,len(col_values)])

        ## Generate predictions
        for row_idx,row in enumerate(data):
            y_pred = gen_model_pred(model, row, column_num, col_values)
            if normalize_loc=='start':
                y_pred = y_pred - y_pred[0]
            if normalize_loc=='end':
                y_pred = y_pred - y_pred[-1]
            if (type(normalize_loc)==int and normalize_loc>=0 and normalize_loc<resolution):
                y_pred = y_pred - y_pred[normalize_loc]
            out_matrix[row_idx,:] = y_pred
        results[column_name] = (col_values, out_matrix)
    return results



def feature_effect_summary(results, kind="boxh", ax=None, num_features=20, **kwargs):
    '''This function plots a comparison of the effects of different features in a predictive model.

    The features are ranked by their "median" effect across a range of data points, where the "effect"
    is measured by the "peak to trough" distance that occurs as that feature varies across its possible range.
    '''
    ## Convert Pandas DataFrame to nparray explicitly to make life easier
    #print('hello!!!')
    columns = list(results.keys())
    data = [importance_distribution_of_variable(results[col_name][1]) for col_name in columns]
    sortind = np.argsort([np.median(d) for d in data])
    if num_features and num_features > 0:
        num_features = min(num_features, len(columns))
    else:
        num_features = len(columns)
    data = [data[idx] for idx in sortind][:num_features]

    if ax is None:
        ax = _gca()
        fig = ax.get_figure()
        fig.set_figwidth(10)
        fig.set_figheight(max(6, math.ceil(num_features*0.5)))
    ax.boxplot(data, notch=0, sym='+', vert=0, whis=1.5)
    ax.set_yticklabels([columns[idx] for idx in sortind]);


def feature_dependence_plots(results, data=None, pts_selected='sample', num_pts=5, figsize=None):
    '''This function visualizes the effect of a single variable in models with complicated dependencies.
    Given a dataset, it will select points in that dataset, and then change the select column across
    different values to view the effect of the model prediction given that variable.
    '''
    ## Convert Pandas DataFrame to nparray explicitly to make life easier
    #print('hello!!!')
    import matplotlib.pyplot as plt

    columns = sorted(list(results.keys()))
    num_rows = len(results[columns[0]][1])  # Get number of sample rows
    row_indexes = np.random.choice(np.arange(num_rows), num_pts)

    n_cols = min(3, len(columns))
    n_rows = math.ceil(len(columns) / n_cols)
    figsize = (n_cols * 4, n_rows * 4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for col_name, ax in zip(columns, axes.flatten()):
        x = results[col_name][0]
        y_values = results[col_name][1][row_indexes]
        for y in y_values:
            ax.plot(x, y)
        ax.set_title(col_name)
    plt.tight_layout()
    return row_indexes


def importance_distribution_of_variable(model_result_array):
    max_result_vec = np.array(list(map(np.max,model_result_array)))
    min_result_vec = np.array(list(map(np.min,model_result_array)))
    return max_result_vec - min_result_vec



