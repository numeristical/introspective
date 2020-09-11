import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta
from sklearn.metrics import precision_recall_curve

def plot_pr_curve(truth_vec, score_vec,
                  x_axis='precision', **kwargs):
    prec, rec, _ = precision_recall_curve(truth_vec,score_vec)
    if x_axis=='precision':
        plt.plot(prec[:-1], rec[:-1], **kwargs)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
    else:
        plt.plot(rec[:-1], prec[:-1], **kwargs)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
    plt.xlim([0,1])
    plt.ylim([0,1])

def plot_pr_curves(truth_vec_list, score_vec_list,
                   x_axis='precision', **kwargs):
    for i in range(len(truth_vec_list)):
        prec, rec, _ = precision_recall_curve(truth_vec_list[i],
                                              score_vec_list[i])
        if x_axis=='precision':
            plt.plot(prec[:-1], rec[:-1], **kwargs)
            plt.xlabel('Precision')
            plt.ylabel('Recall')
        else:
            plt.plot(rec[:-1], prec[:-1], **kwargs)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
        plt.xlim([0,1])
        plt.ylim([0,1])


def histogram_pair(value_vec, binary_vec, bins, smoothing_const=.01,
                   prior_prob=.5, rel_risk=False, error_bar_alpha=.05,
                   figsize = (12,6), **kwargs):
    """Plot the relationship between a numerical feature and a binary outcome.

    This will create two plots stacked vertically.  The upper plot
    is a stacked histogram showing the the counts of 0 and 1 in each
    respective bin.

    The lower plot shows the marginal empirical probability of being a 1
    given that the numerical feature is in a particular value range.

    This gives a simple way to assess the relationship between the
    two variables, especially if it is non-linear. Error bars are also
    shown to demonstrate the confidence of the empirical probability
    (based on the Beta distribution)

    Parameters
    ----------

    value_vec : array-like (containing numerical values)
        The array of numerical values that we are exploring

    binary_vec : array_like (containing 0/1 values)
        The array of binary values that we are exploring

    bins : list or numpy array
        The bin endpoints to use, as if constructing a histogram.

    smoothing_const : float, default = .01
        To avoid issues when a bin contains few or no data points,
        we add in a small number of both positive and negative
        observations to each bin. This controls the weight of the
        added data.

    prior_prob : float, default = .5
        The prior probability reflected by the added data.

    rel_risk : bool, default is False
        If True, this will plot log(emp_prob/prior_prob) rather
        on the y-axis rather than emp_prob.

    error_bar_alpha : float default=.05
        The alpha value to use for the error bars (based on
        the Beta distribution).  Default is 0.05 corresponding
        to a 95% confidence interval.

    figsize : tuple of 2 floats, default=(12,6)
        The size of the "canvas" to use for plotting.

    **kwargs : other
        Other parameters to be passed to the plt.hist command.
    """
    nan_mask = np.isnan(value_vec)
    num_nans = np.sum(nan_mask)
    if num_nans > 0:
        nan_binary_vec = binary_vec[nan_mask]
        binary_vec = binary_vec[~nan_mask]
        value_vec = value_vec[~nan_mask]
        nan_avg_value = np.mean(nan_binary_vec)
        reg_avg_value = np.mean(binary_vec)
    out0 = plt.hist(value_vec[binary_vec == 0], bins=bins, **kwargs)
    out1 = plt.hist(value_vec[binary_vec == 1], bins=bins, **kwargs)
    plt.close()
    plt.figure(figsize=figsize)
    plt.subplot(2, 1, 1)
    plt.hist((value_vec[binary_vec == 0],value_vec[binary_vec == 1]),
              stacked=True, bins=bins, **kwargs)
    bin_leftpts = (out1[1])[:-1]
    bin_rightpts = (out1[1])[1:]
    default_bin_centers = (bin_leftpts + bin_rightpts) / 2
    digitized_value_vec = np.digitize(value_vec, bins)
    bin_centers = np.array([np.mean(value_vec[digitized_value_vec==i])
                                if i in np.unique(digitized_value_vec)
                                else default_bin_centers[i-1]
                                for i in np.arange(len(bins)-1)+1])
    prob_numer = out1[0]
    prob_denom = out1[0] + out0[0]
    smoothing_const = .001
    probs = ((prob_numer + prior_prob * smoothing_const) /
             (prob_denom + smoothing_const))
    plt.subplot(2, 1, 2)
    if rel_risk:
        plt.plot(bin_centers, np.log(probs / prior_prob), '-o')
        plt.xlim(bin_leftpts[0], bin_rightpts[-1])
    else:
        plt.plot(bin_centers[:len(probs)], probs, '-o')
        plt.xlim(bin_leftpts[0], bin_rightpts[-1])
        yerr_mat_temp = beta.interval(1-error_bar_alpha,out1[0]+1,out0[0]+1)
        yerr_mat = np.vstack((yerr_mat_temp[0],yerr_mat_temp[1])) - probs
        yerr_mat[0,:] = -yerr_mat[0,:]
        plt.errorbar(bin_centers[:len(probs)], probs,
                     yerr=yerr_mat, capsize=5)
        plt.xlim(bin_leftpts[0], bin_rightpts[-1])
        if num_nans > 0:
            plt.hlines(y=nan_avg_value, xmin=bin_leftpts[0],
                       xmax=bin_leftpts[1], linestyle='dotted')
            plt.hlines(y=reg_avg_value, xmin=bin_leftpts[0],
                       xmax=bin_leftpts[1], linestyle='dashed')
    return {'bin_centers': bin_centers, 'probs': probs,
            'prob_numer': prob_numer, 'prob_denom': prob_denom}

def ice_plot(model, base_data, column_names, range_pts,
             show_base_pt=True, show_nan=False,
             pred_fn='predict_proba', class_num=1, figsize='auto',
             plots_per_row=3, y_scaling='none'):
    """Generates an ICE plot for a model and data points.

    ICE (Individual Conditional Expectation) plots are a tool for
    understanding model predictions. Given a model, a 'base' data point,
    a column, and a range of values, it will plot the model prediction
    as we change the selected column across the range of values (holding
    all other variables constant).

    This function was developed for use with Jupyter notebooks using
    the 'inline' option for matplotlib.

    Parameters
    ----------

    model : model object
        The model whose behavior we are examining.  Most models are
        supported as long as they are equipped with a "predict" or 
        "predict_proba" method.

    base_data : pandas Series or DataFrame
        The point or points to use as the "base". One curve will be
        plotted for each row in the DataFrame.  If a Series is given,
        just one curve will be plotted.

    column_names : str or iterable of strings
        The names of the columns for which to make ICE plots

    range_pts : list/numpy array or dict
        The range of values to plot for the designated column.  If
        column_names is a string, then range pts can be a list or
        array. If column_names contains multiple columns, then
        range_pts must be a dictionary with the column names as keys
        and the ranges as the corresponding values.  Such a dict
        can be easily created with the `get_range_dict` function
        and subsequently modified if desired.

    show_base_pt : bool, default is True
        Whether or not to show the original value of the data pt
        for the column in question. This is often useful for
        understanding the areas of the curve that correspond to
        realistic values, given the other feature values.

    show_nan : bool, default is False
        If set to true, this will include a np.nan in the range
        of values to plot. This is useful for understanding what
        predictions the model is making in the presence of missing
        data.  The value will be designated by a dotted line on the
        left of the plot.

    pred_fn : 'predict' or 'predict_proba', default is 'predict_proba'
        Which method of the model to be used to generate the values.
        Generally 'predict' should be used for regression and 
        'predict_proba' for classification.

    class_num : int, default is 1
        Which class to get the probability of.  Only relevant when 
        pred_fn = 'predict_proba'.  Default is 1, which gives the
        expected behavior for binary classification.  For multi-class
        classification, changing this number allows the exploration
        of the various class probabilities.

    figsize : 'auto' or tuple of 2 ints, default is 'auto'
        The size of the overall canvas created for multiple plots.
        By default, this will try to choose a reasonable size, but
        can be customized.

    plots_per_row : int, default is 3
        How many plots to put in a single row, when plotting multiple
        columns.

    y_scaling : 'none', 'logit', default is 'none'
        If set to 'logit', the model output will be converted to log odds
        scale before plotting.  This is often useful when working
        with probabilities, particularly small ones.
 """
    if type(base_data)==pd.core.series.Series:
        base_data = pd.DataFrame(base_data).swapaxes('index', 'columns')
    if type(column_names)==str:
        num_plots = 1
        if type(range_pts)!=dict:
            range_pts = {column_names: range_pts}
        column_names = [column_names]
    else:
        num_plots = len(column_names)
    
    plots_per_row = int(np.minimum(num_plots, plots_per_row))
    num_fig_rows = int(np.ceil(num_plots/plots_per_row))
    if ((type(figsize)==str) and (figsize=='auto')):
        figsize=(4*plots_per_row, 3*num_fig_rows)
    plt.figure(figsize=figsize)
    for i,column in enumerate(column_names):
        plt.subplot(num_fig_rows, plots_per_row, i+1)
        for dr in base_data.iterrows():
            data_row = dr[1]
            rp = range_pts[column]
            pred_df = pd.DataFrame(columns=data_row.index)
            pred_df.loc[0] = data_row.copy()
            pred_df = pd.concat([pred_df]*len(rp), ignore_index=True)
            base_pt_df = pred_df.iloc[0:1,:].copy()
            base_pt_df[column] = data_row[column]
            pred_df[column] = rp
            if show_nan:
                new_df = pd.DataFrame(columns=data_row.index)
                new_dr = data_row.copy()
                new_dr[column] = np.nan
                new_df.loc[0] = new_dr
                pred_df = pd.concat((pred_df, new_df))
            if pred_fn=='predict_proba':
                pred_vals = model.predict_proba(pred_df)[:,class_num]
                base_pt_val = model.predict_proba(base_pt_df)[:,class_num]
            if pred_fn=='predict':
                pred_vals = model.predict(pred_df)
                base_pt_val = model.predict(base_pt_df)
            if y_scaling=='logit':
                pred_vals = np.maximum(pred_vals,1e-16)
                pred_vals = np.minimum(pred_vals,1-1e-16)
                pred_vals = np.log(pred_vals/(1-pred_vals))
                base_pt_val = np.maximum(base_pt_val,1e-16)
                base_pt_val = np.minimum(base_pt_val,1-1e-16)
                base_pt_val = np.log(base_pt_val/(1-base_pt_val))
            p = plt.plot(rp, pred_vals[:len(rp)])
            if show_base_pt:
                plt.scatter(base_pt_df[column],base_pt_val)
            plt.title(column)
            if show_nan:
                plt.hlines(pred_vals[-1],
                           xmin=rp[0],
                           xmax=rp[int(np.floor(len(rp)/4+1))],
                           linestyle='dotted', color=p[0].get_color())

def get_range_dict(df_in, max_pts=200):
    """Get reasonable ranges for all columns for ICE plot.

    This function looks at a dataframe, and generates numpy arrays
    for each column based on the min.

    Parameters
    ----------

    df_in : DataFrame
        A dataframe containing the relevant columns.

    max_pts : int, default is 200
        For both numerical and categorical (string) data,  if the 
        number of unique values is less than max_pts, it will choose
        all unique values.  Otherwise, for numerical data it will
        choose max_pts values linearly spaced between the min and max
        values in df and for categorical (string) data, it will choose
        the max_pts most common unique values.


    Returns
    -------

    rd : dict
        A dictionary with the column names as keys and numpy arrays as
        values. This dictionary can then be used as the "range_dict"
        in the `ice_plot` function

    """
    rd = {}
    for col in df_in.columns:
        if df_in.dtypes[col] == 'O':
            rd[col] = np.array(df_in[col].value_counts().index[:max_pts])
        else:
            unique_vals = np.unique(df_in[col])
            if len(unique_vals) < max_pts:
                rd[col] = unique_vals
            else: 
                rd[col] = np.linspace(np.min(df_in[col]),np.max(df_in[col]),max_pts) 
    return rd
