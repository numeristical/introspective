import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta, binom
from sklearn.metrics import precision_recall_curve
from sklearn.base import clone

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

def get_stratified_foldnums(y, num_folds, random_state=42):
    """Given an outcome vector y, assigns each data point to a fold in a stratified manner.
    
    Assumes that y contains only integers between 0 and num_classes-1
    """
    np.random.seed(random_state)
    fn_vec = -1 * np.ones(len(y))
    for y_val in np.unique(y):
        curr_yval_indices = np.where(y==y_val)[0]
        np.random.shuffle(curr_yval_indices)
        index_indices = np.round((len(curr_yval_indices)/num_folds)*np.arange(num_folds+1)).astype(int)
        for i in range(num_folds):
            fold_to_assign = i if ((y_val%2)==0) else (num_folds-i-1)
            fn_vec[curr_yval_indices[index_indices[i]:index_indices[i+1]]] = fold_to_assign
    return(fn_vec)

def cv_predictions(model, X, y, num_cv_folds=5, stratified=True, clone_model=False, random_state=42):
    """Creates a vector of cross-validated predictions given the model and data.

   This function takes a model and repeatedly fits it on all but one fold and
   then makes predictions (using `predict_proba`) on the remaining fold.  It
   returns the full set of cross-validated predictions.

    Parameters
    ----------
    model: The model to be used for the fit and predict_proba calls.  If clone_model
        is True, model will be copied before it is refit, and the original will not 
        be modified.  If clone_model is False, model will be refit and changed.
        The `clone_model` option may not work outside of sklearn.

    X: The feature matrix to be used for the cross-validated predictions

    y: The outcome vector to be used for cross-validated predictions.  Should
        contain integers from 0 to num_classes-1.

    num_cv_folds: The number of folds to create when doing the cross-validated
        fit and predict calls.  More folds will take more time but may yield 
        better results.  Default is 5.

    stratified: Boolean variable indicating whether or not to assign points
        to folds in a stratified manner.  Default is True.

    clone_model: Whether to use the sklearn "clone" function to copy the model
        before it is refit.  If False, the model object will be modified.  The 
        setting True may not work outside of sklearn.  In this case it is
        best to make an identical (before fitting) model object and pass that
        as the argument.

    random_state: A random_state to pass to the fold selection.

    Returns
    ---------------------

    A matrix of size (nrows, ncols) where nrows is the number of rows in X and
    ncols is the number of classes as indicated by y.
    """
    if stratified:
        foldnum_vec = get_stratified_foldnums(y, num_cv_folds, random_state)
    else:
        foldnum_vec = np.floor(np.random.uniform(size=X.shape[0])*num_cv_folds).astype(int)
    model_to_fit = clone(model) if clone_model else model
    n_classes = np.max(y).astype(int)+1
    out_probs = np.zeros((X.shape[0],n_classes))
    for fn in range(num_cv_folds):
        X_tr = X.loc[foldnum_vec!=fn]
        y_tr = y[foldnum_vec!=fn]
        X_te = X.loc[foldnum_vec==fn]
        model_to_fit.fit(X_tr, y_tr)
        out_probs[foldnum_vec==fn,:] = model_to_fit.predict_proba(X_te)
    
    return(out_probs)

def get_stratified_foldnums(y, num_folds, random_state=42):
    """Given an outcome vector y, assigns each data point to a fold in a stratified manner.
    
    Assumes that y contains only integers between 0 and num_classes-1
    """
    np.random.seed(random_state)
    fn_vec = -1 * np.ones(len(y))
    for y_val in np.unique(y):
        curr_yval_indices = np.where(y==y_val)[0]
        np.random.shuffle(curr_yval_indices)
        index_indices = np.round((len(curr_yval_indices)/num_folds)*np.arange(num_folds+1)).astype(int)
        for i in range(num_folds):
            fold_to_assign = i if ((y_val%2)==0) else (num_folds-i-1)
            fn_vec[curr_yval_indices[index_indices[i]:index_indices[i+1]]] = fold_to_assign
    return(fn_vec)

def plot_reliability_diagram(y,
                             x,
                             bins=np.linspace(0,1,21),
                             show_baseline=True,
                             baseline_color="black",
                             baseline_width=1,
                             error_bars=True,
                             error_bar_color='C0',
                             error_bar_alpha=.05,
                             error_bar_width=2,
                             ci_ref="axis",
                             marker=".",
                             marker_color='C1',
                             marker_edge_color="C1",
                             marker_size=50,
                             scaling='none', 
                             scaling_eps=.0001,
                             scaling_base=10, 
                             cap_width=1,
                             cap_size=5,
                             show_histogram=False,
                             bin_color="C0",
                             bin_edge_color="black",  
                             ax1_x_title="Predicted",
                             ax1_y_title="Empirical",
                             ax2_x_title="Predicted Scores",
                             ax2_y_title="Count",
                             ax_title_weight="normal",
                             ax_title_size=12,
                             title_size=16,
                             title_weight='normal',
                             reliability_title="Reliability Diagram",
                             histogram_title="Probability Distribution",
                             layout_pad=3.0,
                             legend_names=['Perfect', 'Model', '95% CI'],
                             legend_size='small',
                             grid_color="#EEEEEE",
                             grid_line_width=0.8,
                             plot_style=None,
                             **kwargs):
    """Plots a reliability diagram of predicted vs empirical probabilities.
    
    Parameters
    ----------
    y: Array-like, length (n_samples). The true outcome values as integers (0 or 1)
    
    x: The predicted probabilities, between 0 and 1 inclusive.
    
    bins: Array-like, the endpoints of the bins used to aggregate and estimate the
        empirical probabilities.  Default is 20 equally sized bins.
        from 0 to 1, i.e. [0,0.05,0.1,...,.95, .1].
        
    show_baseline: Whether or not to print a dotted line representing
        y=x (perfect calibration).  Default is True.
        
    baseline_color: The color of the baseline. Default is black.
    
    baseline_width: The width of the baseline. Default is 1.
    
    error_bars: Whether to show error bars reflecting the confidence
        interval under the assumption that the input probabilities are
        perfectly calibrated. Default is True.
        
    error_bar_color: The color of the errorbar. Default is 'C0', matplotlib blue.   
    error_bar_alpha: The alpha value to use for the error_bars.  Default
        is .05 (a 95% CI).  Confidence intervals are based on the exact
        binomial distribution, not the normal approximation.
        
    error_bar_width: The width of the error bar lines. Default is 2.

    ci_ref: The confidence interval point of reference. If 'axis', the
        confidence interval will be computed around the null hypothesis of
        perfect calibration. If 'point', the confidence intervals will be
        computed around the estimated probability. Default is 'axis'.

    marker: The style of the marker. Default is '.'
    
    marker_color: The color of the marker. Default is 'C1', matplotlib orange.
    
    marker_size: The size of the marker. Default is 50.
    
    scaling: Default is 'none'. Alternative is 'logit' which is useful for
        better examination of calibration near 0 and 1.  Values shown are
        on the scale provided and then tick marks are relabeled.
        
    scaling_eps: Default is .0001.  Ignored unless scaling='logit'. This 
        indicates the smallest meaningful positive probability you
        want to consider.
        
    scaling_base: Default is 10. Ignored unless scaling='logit'. This
        indicates the base used when scaling back and forth.  Matters
        only in how it affects the automatic tick marks.
        
    cap_size: The length of the error bar caps in points. Default is 5.
    
    show_histogram: Whether or not to show a separate histogram of the
        number of values in each bin.  Default is False.
        
    bin_color: The color of the histogram bins. Default is 'C0', 
        matplotlib blue.
    
    bin_edge_color: The color of the edges around the histogram bins. 
        Default is 'black'.
    
    ax1_x_title: X-axis title for reliability plot. Default is 
        "Predicted".
    
    ax1_y_title: Y-axis title for reliability plot. Default is 
        "Empirical".
    
    ax2_x_title: X-axis title for histogram. Default is "Predicted 
        Scores".
    
    ax2_y_title: Y-axis title for histogram. Default is "Count".
    
    ax_title_weight: The font weight for axes titles. Default 
        is "normal".
    
    ax_title_size: The font size for the axes titles. Default 
        is 12.
    
    title_size: The font size for the subplot titles. Default 
        is 16.
    
    title_weight: The font weight for the subplot titles. Default 
        is 'normal'.
    
    reliability_title: The title for the reliability plot. Default 
        is "Reliability Diagram".
    
    histogram_title: The title for the histogram. Default is "Probability 
        Distribution".
    
    layout_pad: Space to add between subplots to give y-axis title and 
        labels room to breath. Default is 3.0.
        
    legend_names: List of names for the legend labels. Defaults to 
        'Perfect', 'Model', '95% CI'.
    
    legend_size: 'xx-small', 'x-small', 'small', 'medium', 
        'large', 'x-large', 'xx-large' or integer for the legend 
        size. Defaults to 'small'.
        
    grid_color: The color of the grid. Default is "#EEEEEE".
    
    grid_line_width: The width of the gridlines. Default is 0.8.
    
    plot_style: Check available styles "plt.style.available".
        ['default', 'classic', 'Solarize_Light2', '_classic_test_patch', 'bmh', 
        'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 
        'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 
        'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 
        'seaborn-notebook', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 
        'seaborn-ticks', 'seaborn-white','seaborn-whitegrid', 'tableau-colorblind10'] 
        Defaults to None.
        
    **kwargs: additional args to be passed to the plt.scatter matplotlib call.
    
    Returns
    -------
    A dictionary containing the x and y points plotted (unscaled) and the 
        count in each bin.
    """
    
    # Set Plot Style
    if plot_style is None:
        None
        
    else:
        plt.style.use(plot_style)

    digitized_x = np.digitize(x, bins)
    mean_count_array = np.array([[np.mean(y[digitized_x == i]),
                                  len(y[digitized_x == i]),
                                  np.mean(x[digitized_x==i])] 
                                  for i in np.unique(digitized_x)])
    x_pts_to_graph = mean_count_array[:,2]
    y_pts_to_graph = mean_count_array[:,0]
    bin_counts = mean_count_array[:,1]
    if show_histogram:
        plt.subplot(1,2,1)
    if scaling=='logit':
        x_pts_to_graph_scaled = my_logit(x_pts_to_graph, eps=scaling_eps,
                                         base=scaling_base)
        y_pts_to_graph_scaled = my_logit(y_pts_to_graph, eps=scaling_eps,
                                         base=scaling_base)
        prec_int = np.max([-np.floor(np.min(x_pts_to_graph_scaled)),
                    np.ceil(np.max(x_pts_to_graph_scaled))])
        prec_int = np.max([prec_int, -np.floor(np.log10(scaling_eps))])
        low_mark = -prec_int
        high_mark = prec_int
        if show_baseline:
            plt.plot([low_mark, high_mark], [low_mark, high_mark],'--', color=baseline_color, linewidth=baseline_width, zorder=2)
        plt.scatter(x_pts_to_graph_scaled, 
                    y_pts_to_graph_scaled,
                    c=marker_color,
                    ec=marker_edge_color,
                    s=marker_size, 
                    zorder=3, 
                    marker=marker, 
                    **kwargs)
        locs, labels = plt.xticks()
        labels = np.round(my_logistic(locs, base=scaling_base), decimals=4)
        plt.xticks(locs, labels)
        locs, labels = plt.yticks()
        labels = np.round(my_logistic(locs, base=scaling_base), decimals=4)
        plt.yticks(locs, labels)
        plt.grid(which='major', color=grid_color, linewidth=grid_line_width, zorder=1)
        plt.legend(legend_names, loc='upper left', fontsize=legend_size)
        if error_bars:
            if ci_ref == "axis":
                prob_range_mat = binom.interval(1-error_bar_alpha,bin_counts,x_pts_to_graph)/bin_counts
                yerr_mat = (my_logit(prob_range_mat, eps=scaling_eps, base=scaling_base) -
                           my_logit(x_pts_to_graph, eps=scaling_eps, base=scaling_base))
                yerr_mat[0,:] = -yerr_mat[0,:]
                plt.errorbar(x_pts_to_graph_scaled,
                             x_pts_to_graph_scaled,
                             elinewidth=error_bar_width,
                             ecolor=error_bar_color,
                             yerr=yerr_mat,
                             capthick=cap_width,
                             capsize=cap_size,
                             ls="none",
                             zorder=2)
            elif ci_ref == "point":
                prob_range_mat = binom.interval(1-error_bar_alpha,bin_counts,y_pts_to_graph)/bin_counts
                yerr_mat = (my_logit(prob_range_mat, eps=scaling_eps, base=scaling_base) -
                           my_logit(y_pts_to_graph, eps=scaling_eps, base=scaling_base))
                yerr_mat[0,:] = -yerr_mat[0,:]
                plt.errorbar(x_pts_to_graph_scaled,
                             y_pts_to_graph_scaled,
                             elinewidth=error_bar_width,
                             ecolor=error_bar_color,
                             yerr=yerr_mat,
                             capthick=cap_width,
                             capsize=cap_size,
                             ls="none",
                             zorder=2)
            plt.legend(['y=x', 'Model', '95% CI'], loc='upper left', fontsize=legend_size)
        plt.axis([low_mark-.1, high_mark+.1, low_mark-.1, high_mark+.1])
        plt.grid(which='major', color=grid_color, linewidth=grid_line_width, zorder=1)
        plt.legend(legend_names, loc='upper left')
    if scaling!='logit':
        if show_baseline:
            plt.plot(np.linspace(0,1,100),(np.linspace(0,1,100)),'--', color=baseline_color, linewidth=baseline_width, zorder=2)
        # for i in range(len(y_pts_to_graph)):
        plt.scatter(x_pts_to_graph,
                    y_pts_to_graph, 
                    c=marker_color,
                    ec=marker_edge_color,
                    s=marker_size, 
                    zorder=4, 
                    marker=marker, 
                    **kwargs)
        plt.axis([-0.1,1.1,-0.1,1.1])
        plt.grid(which='major', color=grid_color, linewidth=grid_line_width, zorder=1)
        plt.legend(legend_names, loc='upper left', fontsize=legend_size)
        if error_bars:
            if ci_ref == "axis":
                yerr_mat = binom.interval(1-error_bar_alpha,bin_counts,x_pts_to_graph)/bin_counts - x_pts_to_graph
                yerr_mat[0,:] = -yerr_mat[0,:]
                plt.errorbar(x_pts_to_graph,
                             x_pts_to_graph,
                             elinewidth=error_bar_width,
                             ecolor=error_bar_color,
                             yerr=yerr_mat,
                             capthick=cap_width,
                             capsize=cap_size,
                             ls="none",
                             zorder=3)
            elif ci_ref == "point":
                yerr_mat = binom.interval(1-error_bar_alpha,bin_counts,y_pts_to_graph)/bin_counts - y_pts_to_graph
                yerr_mat[0,:] = -yerr_mat[0,:]
                plt.errorbar(x_pts_to_graph,
                             y_pts_to_graph,
                             elinewidth=error_bar_width,
                             ecolor=error_bar_color,
                             yerr=yerr_mat,
                             capthick=cap_width,
                             capsize=cap_size,
                             ls="none",
                             zorder=3)
    plt.xlabel(ax1_x_title, fontsize=ax_title_size, fontweight=ax_title_weight)
    plt.ylabel(ax1_y_title, fontsize=ax_title_size, fontweight=ax_title_weight)
    plt.title(reliability_title, fontsize=title_size, fontweight=title_weight)
    plt.grid(which='major', color=grid_color, linewidth=grid_line_width, zorder=1)
    plt.legend(legend_names, loc='upper left', fontsize=legend_size)
    if show_histogram:
        plt.subplot(1,2,2)
        plt.hist(x,
                 bins=bins, 
                 ec=bin_edge_color,
                 color=bin_color,
                 zorder=2)
        plt.xlabel(ax2_x_title, fontsize=ax_title_size, fontweight=ax_title_weight)
        plt.ylabel(ax2_y_title, fontsize=ax_title_size, fontweight=ax_title_weight)
        plt.title(histogram_title, fontsize=title_size, fontweight=title_weight)
        plt.grid(which='major', color=grid_color, linewidth=grid_line_width, zorder=1)
        plt.tight_layout(pad=layout_pad)
    out_dict = {}
    out_dict['pred_probs'] = x_pts_to_graph
    out_dict['emp_probs'] = y_pts_to_graph
    out_dict['bin_counts'] = bin_counts
    return(out_dict)

def my_logit(vec, base=np.exp(1), eps=1e-16):
    vec = np.clip(vec, eps, 1-eps)
    return (1/np.log(base)) * np.log(vec/(1-vec))

def my_logistic(vec, base=np.exp(1)):
    return 1/(1+base**(-vec))

