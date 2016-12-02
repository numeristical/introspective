import math
import warnings
import numpy as np
import pandas as pd
from .utils import _gca, is_classifier, is_regressor


class ModelXRay(object):
    """This class executes a model over a broad range of modified data points to analyze aspects of its performance.

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
    """

    def __init__(self, model, data, columns=None, resolution=100, normalize_loc=None):

        self.model = model
        self.data = data

        if type(data) == pd.DataFrame:
            self.data_values = data.values
        else:
            self.data_values = data

        self.columns = columns
        self.results = self._model_xray(columns, resolution, normalize_loc)


    def _get_data_rows(self, row_nums):
        if type(self.data) == pd.DataFrame:
            return self.data.iloc[row_nums]
        else:
            return self.data[row_nums, :]


    def _get_predictions(self, rows):
        # Catch deprecated warnings from Predict call
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if is_classifier(self.model):
                y_pred = self.model.predict_proba(rows)[:,1]
            else:
                y_pred = self.model.predict(rows)
        return y_pred


    def gen_model_pred(self, row, col_idx, values):
        rows = []
        for val in values:
            sim_row = row.copy()
            sim_row[col_idx] = val
            rows.append(sim_row)
        # If the row is a Series, make it into a DF
        if type(rows[0]) == pd.Series:
            rows = pd.DataFrame(rows)
        y_pred = self._get_predictions(rows)
        return y_pred

    def _model_xray(self, columns, resolution, normalize_loc):
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
            if type(self.data) == pd.DataFrame:
                columns = self.data.columns
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
        if type(self.data) == pd.DataFrame:
            for column in columns:
                try:
                    column_nums.append(self.data.columns.get_loc(column))
                except KeyError:
                    ## TODO
                    pass
        else:
            # Column Index and Column Names are the same
            if type(columns[0]) == int:
                column_nums = columns
            else:
                column_nums = range(len(columns))

        # Use the Numpy array of data values to ease indexing by col. numbers
        results = {}
        num_pts = len(self.data_values)
        for column_num, column_name in zip(column_nums, columns):
            if (len(np.unique(self.data_values[:,column_num])) > resolution):
                col_values = np.linspace(np.min(self.data_values[:,column_num]),
                    np.max(self.data_values[:,column_num]),resolution)
            else:
                col_values = np.sort(np.unique(self.data_values[:,column_num]))
            ## Define the empty data structure to output
            out_matrix = np.zeros([num_pts,len(col_values)])

            ## Generate predictions
            if type(self.data) == pd.DataFrame:
                rows = self.data.iterrows()
            else:
                rows = enumerate(self.data)
            for loop_idx, (row_idx, row) in enumerate(rows):
                y_pred = self.gen_model_pred(row, column_num, col_values)
                if normalize_loc=='start':
                    y_pred = y_pred - y_pred[0]
                if normalize_loc=='end':
                    y_pred = y_pred - y_pred[-1]
                if (type(normalize_loc)==int and normalize_loc>=0 and normalize_loc<resolution):
                    y_pred = y_pred - y_pred[normalize_loc]
                out_matrix[loop_idx,:] = y_pred
            results[column_name] = (col_values, out_matrix)
        return results


    def feature_effect_summary(self, kind="boxh", num_features=20, ax=None):
        '''This function plots a comparison of the effects of different features in a complex predictive model.

        In more complicated predictive models, the effect of an individual feature can be highly dependent on the values
        of the other features.  It could be that a feature has a large effect in one context but a negligible effect in another.
        This visualization attempts to shed light on the range of possibilities of the effect of a feature, by giving a boxplot
        showing the range of possibilities of the effect of a feature.

        The features are ranked by their "median" effect across a range of data points, where the "effect"
        is measured by the "peak to trough" distance that occurs as that feature varies across its possible range.

            Parameters
            ----------

            results : This is a results object from a call to model_xray.  Ideally, the model_xray was given a reasonably large amount of data
                so that we can empirically see a broad range of possibilities.

            kind : Currently only 'boxh' (horizontal boxplot) is supported

            ax : If desired, a particular axis on which to generate the plot can be passed to the function

            num_features : This specifies the maximum number of features to include in the boxplot.  The function chooses the most significant
                features as measured by the median peak-to-trough effect size.

            Returns
            -------
        '''
        ## Convert Pandas DataFrame to nparray explicitly to make life easier

        columns = list(self.results.keys())
        result_data = [importance_distribution_of_variable(self.results[col_name][1]) for col_name in columns]
        sortind = np.argsort([np.median(d) for d in result_data])
        if num_features and num_features > 0:
            num_features = min(num_features, len(columns))
        else:
            num_features = len(columns)
        plot_data = [result_data[idx] for idx in sortind][-num_features:]

        if ax is None:
            ax = _gca()
            fig = ax.get_figure()
            fig.set_figwidth(10)
            fig.set_figheight(max(6, int(math.ceil(num_features*0.5))))
        ax.boxplot(plot_data, notch=0, sym='+', vert=0, whis=1.5)
        ax.set_yticklabels([columns[idx] for idx in sortind][-num_features:]);


    def feature_dependence_plots(self, show_base_points=True, pts_selected='sample', num_pts=5, figsize=None):
        '''This function visualizes the effect of a single variable in models with complicated dependencies.
        Given a dataset, it will select points in that dataset, and then change the select column across
        different values to view the effect of the model prediction given that variable. These have been called
        Individual Conditional Expectation plots (or ICE-plots), see Goldstein, Kapelner, Bleich,
        Pitkin. Peeking Inside the Black Box: Visualizing Statistical Learning With Plots of Individual
        Conditional Expectation. Journal of Computational and Graphical Statistics (March 2014)
        '''

        import matplotlib.pyplot as plt

        columns = sorted(list(self.results.keys()))
        num_rows = len(self.results[columns[0]][1])  # Get number of sample rows
        if (type(pts_selected)==str and pts_selected=='sample'):
            row_indexes = np.random.choice(np.arange(num_rows), num_pts)
        else:
            row_indexes = pts_selected

        if show_base_points:
            base_rows = self._get_data_rows(row_indexes)
            y_base_points = self._get_predictions(base_rows)
        else:
            y_base_points = None

        n_cols = min(3, len(columns))
        n_rows = int(math.ceil(len(columns) / n_cols))
        figsize = (n_cols * 4, n_rows * 4)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        for col_name, ax in zip(columns, axes.flatten()):
            x = self.results[col_name][0]
            y_values = self.results[col_name][1][row_indexes]
            for y in y_values:
                ax.plot(x, y)
            # Plot Base Points
            if y_base_points is not None:
                ax.scatter(base_rows[col_name], y_base_points)
            ax.set_title(col_name)
        plt.tight_layout()
        return row_indexes


    def explain_prediction_difference(self, index_1, index_2, tol=.03, verbose=True):
        '''Given the indices of two points in the "xray"-ed data set, this function gives an explanation
        of the factors contributing to the difference in the predictions.

        Starting with the first point given, the considers changing each feature from its current value to that
        possessed by the second point.  The function evaluates the target in both scenarios and determines the
        feature value change that creates the biggest (absolute) change in the target.  This change is selected
        and the current point becomes the new point with the new feature value.  This is repeated until the new
        target value is within a factor of 1+tol of the second point.
        '''
        data_row_1 = self._get_data_rows(index_1)
        data_row_2 = self._get_data_rows(index_2)
        return explain_prediction_difference(self.model, data_row_1, data_row_2, tol, verbose)


def importance_distribution_of_variable(model_result_array):
    max_result_vec = np.array(list(map(np.max,model_result_array)))
    min_result_vec = np.array(list(map(np.min,model_result_array)))
    return max_result_vec - min_result_vec


def explain_prediction_difference(model, data_row_1, data_row_2, tol=.03, verbose=True, decimals = 4):
    '''Given a model and two single row data frames, this function gives an explanation
        of the factors contributing to the difference in the predictions.

        Starting with the first point given, the considers changing each feature from its current value to that
        possessed by the second point.  The function evaluates the target in both scenarios and determines the
        feature value change that creates the biggest (absolute) change in the target.  This change is selected
        and the current point becomes the new point with the new feature value.  This is repeated until the new
        target value is within a factor of 1+tol of the second point.
        '''
    column_names = data_row_1.index
    num_columns = len(column_names)

    dr_1 = data_row_1.values.reshape(1,-1)
    dr_2 = data_row_2.values.reshape(1,-1)
    column_list = list(range(num_columns))
    curr_pt = np.copy(dr_1)
    if is_classifier(model):
        val1 = model.predict_proba(dr_1)[0,1]
        val2 = model.predict_proba(dr_2)[0,1]
    else:
        val1 = model.predict(dr_1)[0]
        val2 = model.predict(dr_2)[0]
    if verbose:
        print('Your initial point has a target value of {}'.format(np.round(val1,decimals=decimals)))
        print('Your final point has a target value of {}'.format(np.round(val2,decimals=decimals)))
    pt_list = [dr_1]
    val_list = [val1]
    curr_val = val1
    final_val = val2
    feat_list =[]
    move_list = []
    feat_val_change_list = []
    #for num_steps in range(4):
    while (((curr_val/final_val) >(1+tol)) or ((curr_val/final_val) <(1-tol))):
        biggest_move = 0
        best_column = -1
        best_val = curr_val
        for i in column_list:
            test_pt = np.copy(curr_pt)
            prev_feat_val = test_pt[0,i]
            subst_val = dr_2[0,i]
            test_pt[0,i] = subst_val
            if is_classifier(model):
                test_val = model.predict_proba(test_pt)[0,1]
            else:
                test_val = model.predict(test_pt)[0]
            move_size = (test_val - curr_val)
            if(np.abs(move_size)>=np.abs(biggest_move)):
                biggest_move = move_size
                best_column = i
                best_val = test_val
                old_feat_val = prev_feat_val
                new_feat_val = subst_val
        subst_val = dr_2[0,best_column]
        curr_pt[0,best_column] = subst_val
        val_list.append(best_val)
        curr_val = best_val
        if verbose:
            print('Changing {} from {} to {}'.format(column_names[best_column],np.round(old_feat_val,decimals=decimals),np.round(new_feat_val,decimals=decimals)))
            print('\t\tchanges your target by {} to {}'.format(np.round(biggest_move,decimals=decimals), np.round(best_val,decimals=decimals)))
            print('----------')
            if not (((curr_val/final_val) >(1+tol)) or ((curr_val/final_val) <(1-tol))):
                print('Tolerance of {} reached'.format(tol))
                print('Current value of {} is within {}% of {}'.format(np.round(curr_val,decimals=decimals),(100*tol),np.round(final_val,decimals=decimals)))
        feat_list.append(column_names[best_column])
        column_list.remove(best_column)
        move_list.append(biggest_move)
        feat_val_change_list.append((old_feat_val, new_feat_val))
    return feat_list, feat_val_change_list, move_list, val_list

def _natural_cubic_spline_basis_expansion(xpts,knots):
    num_knots = len(knots)
    num_pts = len(xpts)
    outmat = np.zeros((num_pts,num_knots))
    outmat[:,0]= np.ones(num_pts)
    outmat[:,1] = xpts
    def make_func_H(k):
        def make_func_d(k):
            def func_d(x):
                denom = knots[-1] - knots[k-1]
                numer = np.maximum(x-knots[k-1],np.zeros(len(x)))**3 - np.maximum(x-knots[-1],np.zeros(len(x)))**3
                return numer/denom
            return func_d
        def func_H(x):
            d_fun_k = make_func_d(k)
            d_fun_Km1 = make_func_d(num_knots-1)
            return d_fun_k(x) -  d_fun_Km1(x)
        return func_H    
    for i in range(1,num_knots-1):
        curr_H_fun = make_func_H(i)
        outmat[:,i+1] = curr_H_fun(xpts)
    return outmat

def prob_calibration_function(scorevec, truthvec,  reg_param_vec='default',
                                knots = 'all', extrapolate=True):
    """This function takes an uncalibrated set of scores and the true 0/1 values and returns a calibration function.

    This calibration function can then be applied to other scores from the same model and will return an accurate probability
    based on the data it has seen.  For best results, the calibration should be done on a separate validation set (not used
    to train the model).
    """
    from sklearn import linear_model
    from sklearn.metrics import log_loss, make_scorer
    
    knot_vec = np.unique(scorevec)    
    X_mat = _natural_cubic_spline_basis_expansion(scorevec, knot_vec)   

    
    if ((type(reg_param_vec)==str) and (reg_param_vec=='default')):
        reg_param_vec = 10**np.linspace(-6,6,241)
    print("Trying {} values of C between {} and {}".format(len(reg_param_vec),np.min(reg_param_vec),np.max(reg_param_vec)))
    reg = linear_model.LogisticRegressionCV(Cs=reg_param_vec, cv=5, scoring=make_scorer(log_loss,needs_proba=True, greater_is_better=False))
    reg.fit(X_mat, truthvec)
    print("Best value found C = {}".format(reg.C_))
            
    def calibrate_scores(new_scores):
        if (not extrapolate):
            new_scores = np.maximum(new_scores,knot_vec[0]*np.ones(len(new_scores)))
            new_scores = np.minimum(new_scores,knot_vec[-1]*np.ones(len(new_scores)))
        basis_exp = _natural_cubic_spline_basis_expansion(new_scores,knot_vec)
        outvec = reg.predict_proba(basis_exp)[:,1]
        return outvec
    return calibrate_scores


