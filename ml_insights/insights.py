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
    '''This function visualizes the effect of a single variable in models with complicated dependencies.
    Given a dataset, it will select points in that dataset, and then change the select column across
    different values to view the effect of the model prediction given that variable.
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


def feature_effect_summary(results, kind="boxh", ax=None, **kwargs):
    '''This function visualizes the effect of a single variable in models with complicated dependencies.
    Given a dataset, it will select points in that dataset, and then change the select column across
    different values to view the effect of the model prediction given that variable.
    '''
    ## Convert Pandas DataFrame to nparray explicitly to make life easier
    #print('hello!!!')
    if ax is None:
        ax = _gca()

    columns = list(results.keys())
    data = [importance_distribution_of_variable(results[col_name][1]) for col_name in columns]
    sortind = np.argsort([np.median(d) for d in data])
    data = [data[idx] for idx in sortind]

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



