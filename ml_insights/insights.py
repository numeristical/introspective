import math
import numpy as np
import pandas as pd
from .utils import _gca

def gen_model_pred(model, row, col_idx, values, is_classification=False):
    rows = []
    for val in values:
        sim_row = row.copy()
        sim_row[col_idx] = val
        rows.append(sim_row)
    if is_classification:
        y_pred = model.predict_proba(rows)[:,1]
    else:
        y_pred = model.predict(rows)
    return y_pred


def dependency_predictions(model, data, columns=None, resolution=100, normalize_loc=None, is_classification=False, **kwargs):
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
        col_values = np.linspace(np.min(data[:,column_num]),np.max(data[:,column_num]),resolution)

        ## Define the empty data structure to output
        out_matrix = np.zeros([num_pts,resolution])

        ## Plot the lines

        for row_idx,row in enumerate(data):
            y_pred = gen_model_pred(model, row, column_num, col_values, is_classification)
            if normalize_loc=='start':
                y_pred = y_pred - y_pred[0]
            if normalize_loc=='end':
                y_pred = y_pred - y_pred[-1]
            if (type(normalize_loc)==int and normalize_loc>=0 and normalize_loc<resolution):
                y_pred = y_pred - y_pred[normalize_loc]
            out_matrix[row_idx,:] = y_pred
        results[column_name] = (col_values, out_matrix)
    return results


def dependency_plot(results, kind="boxh", ax=None, **kwargs):
    '''This function visualizes the effect of a single variable in models with complicated dependencies.
    Given a dataset, it will select points in that dataset, and then change the select column across
    different values to view the effect of the model prediction given that variable.
    '''
    ## Convert Pandas DataFrame to nparray explicitly to make life easier
    #print('hello!!!')
    if ax is None:
        ax = _gca()

    columns = list(results.keys())
    data = [results[col_name][1] for col_name in columns]
    sortind = np.argsort([np.median(d) for d in data])
    data = [data[idx] for idx in sortind]

    ax.boxplot(data, notch=0, sym='+', vert=0, whis=1.5)
    ax.set_yticklabels([columns[idx] for idx in sortind]);


def dependency_lines(results, data=None, pts_selected='sample', num_pts=5, figsize=None):
    '''This function visualizes the effect of a single variable in models with complicated dependencies.
    Given a dataset, it will select points in that dataset, and then change the select column across
    different values to view the effect of the model prediction given that variable.
    '''
    ## Convert Pandas DataFrame to nparray explicitly to make life easier
    #print('hello!!!')
    import matplotlib.pyplot as plt

    columns = list(results.keys())
    num_rows = len(results[columns[0]][1])  # Get number of sample rows
    row_indexes = np.random.choice(np.arange(num_rows), num_pts)

    n_cols = min(3, len(columns))
    n_rows = math.ceil(len(columns) / n_cols)
    figsize = (n_rows * 4, n_cols * 4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for col_name, ax in zip(columns, axes.flatten()):
        x = results[col_name][0]
        y_values = results[col_name][1][row_indexes]
        for y in y_values:
            ax.plot(x, y)
        ax.set_title(col_name)
    plt.tight_layout()
    return row_indexes


def dependence_plot(model, dataset, column_num, pts_selected='sample', num_pts=5, col_values='auto',
                       resolution = 100, show_base_pts=True, normalize_loc='none',classification=False,
                       ax=None, show_plot = True, **kwargs):
    '''This function visualizes the effect of a single variable in models with complicated dependencies.
    Given a dataset, it will select points in that dataset, and then change the select column across
    different values to view the effect of the model prediction given that variable.
    '''
    ## Convert Pandas DataFrame to nparray explicitly to make life easier
    #print('hello!!!')
    if ax is None:
        ax = _gca()

    if type(dataset)==pd.DataFrame:
        dataset = dataset.values

    ## Determine which points to serve as base points depending on pts_selected and num_pts
    if ((type(pts_selected) == str) and (pts_selected=='sample')):
        pts_chosen = np.random.choice(dataset.shape[0],np.minimum(dataset.shape[0],num_pts),replace=False)
    elif ((type(pts_selected) == str) and (pts_selected=='first')):
        pts_chosen = np.array(range(np.minimum(dataset.shape[0],num_pts)))
    else:
        pts_chosen = np.array(pts_selected)

    ## Determine the range of values to plot for the chosen column
    if (type(col_values)==str and col_values=='auto'):
        values = np.linspace(np.min(dataset[:,column_num]),np.max(dataset[:,column_num]),resolution)
    else:
        values = np.array(col_values)

    ## Define the empty data structure to output
    out_matrix = np.zeros([num_pts,resolution])

    ## Plot the lines

    for i,row in enumerate(dataset[pts_chosen,:]):
        if classification:
            y_pred = gen_model_pred(model, row, column_num, values, classification=True)
        else:
            y_pred = gen_model_pred(model, row, column_num, values)
        if normalize_loc=='start':
            y_pred = y_pred - y_pred[0]
        if normalize_loc=='end':
            y_pred = y_pred - y_pred[-1]
        if (type(normalize_loc)==int and normalize_loc>=0 and normalize_loc<resolution):
            y_pred = y_pred - y_pred[normalize_loc]
        if show_plot:
            ax.plot(values, y_pred)
        out_matrix[i,:] = y_pred
    if(show_base_pts and normalize_loc=='none' and show_plot):
        if classification:
            pred_vals = model.predict_proba(dataset[pts_chosen,:])[:,1]
            #ax.scatter(dataset[pts_chosen,column_num],model.predict_proba(dataset[pts_chosen,:])[:,1],**kwargs)
            ax.scatter(dataset[pts_chosen,column_num],pred_vals,**kwargs)
        else:
            pred_vals = model.predict(dataset[pts_chosen,:])
            #ax.scatter(dataset[pts_chosen,column_num],model.predict(dataset[pts_chosen,:]),**kwargs)
            ax.scatter(dataset[pts_chosen,column_num],pred_vals,**kwargs)
    return values, out_matrix

def median_dependence_plot(model, dataset, column_num, pts_selected='sample', num_pts=100, col_values='auto',
                       resolution = 100, ax=None, **kwargs):
    '''This function attempts to characterize the "average" effect (and variation around) when a variable changes value.
    '''
    if ax is None:
        ax = _gca()

    ## Convert Pandas DataFrame to nparray explicitly to make life easier
    if type(dataset)==pd.DataFrame:
        dataset = dataset.values

    ## Determine which points to serve as base points depending on pts_selected and num_pts
    if ((type(pts_selected) == str) and (pts_selected=='sample')):
        pts_chosen = np.random.choice(dataset.shape[0],min(dataset.shape[0],num_pts),replace=False)
    elif ((type(pts_selected) == str) and (pts_selected=='first')):
        pts_chosen = np.array(range(np.min(dataset.shape[0],num_pts)))
    else:
        pts_chosen = np.array(pts_selected)

    ## Determine the range of values to plot for the chosen column
    if (type(col_values)==str and col_values=='auto'):
        values_to_plot = np.linspace(np.min(dataset[:,column_num]),np.max(dataset[:,column_num]),resolution)
    else:
        values_to_plot = np.array(col_values)

    ## Create the 2d array for differences by values
    values_xvec = (values_to_plot[1:]+values_to_plot[:-1])/2
    pred_vals_diff = np.zeros((len(pts_chosen),len(values_xvec)))
    pred_vals = np.zeros((len(pts_chosen),len(values_to_plot)))


    for i,row in enumerate(dataset[pts_chosen,:]):
        y_pred, values = gen_model_pred(model, row, column_num, values_to_plot)
        diffvec = np.diff(y_pred)
        #plt.scatter(values_xvec,diffvec,**kwargs)
        pred_vals_diff[i,:]=diffvec
        pred_vals[i,:]=y_pred

    median_diff_vec = np.zeros(len(values_xvec))
    median_val_vec = np.zeros(len(values_to_plot))

    for j in range(len(values_to_plot)):
        median_val_vec[j] = np.median(pred_vals[:,j])
    for j in range(len(values_xvec)):
        median_diff_vec[j] = np.median(pred_vals_diff[:,j])
    #plt.plot(values_xvec,np.cumsum(median_diff_vec),c='k',marker='_')
    ax.plot(values_to_plot,(median_val_vec),c='k')



    for j in range((pred_vals_diff.shape[0])):
        plt.scatter(values_xvec,pred_vals_diff[j,:]+median_val_vec[1:]-median_diff_vec,**kwargs)

    return pred_vals,pred_vals_diff


def importance_distribution_of_variable(model_result_array):
    max_result_vec = np.array(list(map(np.max,model_result_array)))
    min_result_vec = np.array(list(map(np.min,model_result_array)))
    return max_result_vec - min_result_vec



