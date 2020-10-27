import numpy as np
from .calibration_utils import _natural_cubic_spline_basis_expansion
from .calibration_utils import create_yeqx_bias_vectors, logreg_cv
from .calibration_utils import my_logit, my_logistic
import random
import warnings
import matplotlib.pyplot as plt

class SplineCalib(object):
    """Probability calibration using cubic splines.

    This defines a calibrator object.  The calibrator can be `fit` on 
    model outputs and truth values.  After being fit, it can then be used to
    `calibrate` model outputs.

    This is similar to the sklearn fit/transform paradigm, except that it is 
    intended for post-processing of model outputs rather than preprocessing 
    of model inputs.

    Parameters
    ----------

    penalty : 'l1' or 'l2'
        What kind of coefficient penalization to use.  The default is 'l2',
        which does a standard "smoothing" spline that minimizes the second 
        derivative of the resulting function.  An 'l1' penalty will typically
        give function which has a sparser representation in the spline bases.
        The 'l1' penalty can only be used with the 'liblinear' or 'saga'
        solver and tends to be considerably slower.

    solver : 'lbfgs','liblinear','newton-cg','sag','saga'
        Which solver to use in the sklearn LogisticRegression object that
        powers the spline fitting. Specifying 'default' will use the 'lbfgs'
        for L2 penalty and 'liblinear' for L1.

    knot_sample_size : 
        The number of knots to randomly sample from the training
        values.  More knots take longer to fit.  Too few knots may underfit.
        Too many knots could overfit, but usually the regularization will
        control that from happening. If `knot_sample_size` exceeds the number
        of unique values in the input, then all unique values will be chosen.

    add_knots :
        A list (or np_array) of knots that will be used for the 
        spline fitting in addition to the random sample.  This may be useful
        if you want to force certain knots to be used in areas where the data
        is sparse.

    reg_param_vec :
        A list (or np_array) of values to try for the 'C' parameter
        for regularization in the sklearn LogisticRegression. These should
        be positive numbers on a logarithmic scale for best results.  If
        'default' is chosen it will try 17 evenly spaced values (log scale)
        between .0001 and 10000 (inclusive)

    cv_spline : 
        Number of folds to use for the cross-validation to find the 
        best regularization parameter.  Default is 5.  Folds are chosen
        in a stratified manner.

    random_state : 
        If desired, can specify the random state for the generation
        of the stratified folds.

    unity_prior : 
        If True, routine will add synthetic data along the axis y=x as
        a "prior" distribution that favors that function.  Default is True.

    unity_prior_weight : 
        The total weight of data points added when unity_prior
        is set to True.  Bigger values will force the calibration curve closer
        to the line y=x. Default is 100, meaning the synthetic data in total
        counts as much weight as 100 data points.

    unity_prior_gridsize : 
        The resolution of the grid used to create the
        unity_prior data augmentation.  Default is 100, meaning it would
        create synthetic data at x=0, .01 ,.02 ,...,.99 ,1.

    logodds_scale : 
        Whether or not to transform the x-values to the log odds
        scale before doing the basis expansion.  Default is True and is 
        recommended unless it is suspected that the uncalibrated probabilities
        already have a logistic relationship to the true probabilities.

    logodds_eps : 
        Used only when logodds_scale=True.  Since 0 and 1 map to
        positive and negative infinity on the logodds scale, we must
        specify a minimum and maximum probability before the transformation.
        Default is 'auto' which chooses a reasonable value based on the
        smallest positive value seen and the largest value smaller than 1.

    reg_prec : 
        A positive integer designating the number of decimal places to 
        which to round the log_loss when choosing the best regularization
        parameter. Algorithm breaks ties in favor of more regularization. 
        Higher numbers will potentially use less regularization and lower 
        numbers use more regularization. Default is 4.

    force_knot_endpts : 
        If True, the smallest and largest input value will
        automatically chosen as knots, and `knot_sample_size`-2 knots
        will be chosen among the remaining values. Default is True.


    Attributes
    ----------

    n_classes: The number of classes for which the calibrator was fit.

    knot_vec: The knots chosen (on the probability scale).

    knot_vec_tr: If logodds_scale = True, this will be the values of 
        the knots on the logodds scale.  Otherwise it is the same
        as knot_vec.

    lrobj: (binary) The resulting sklearn LogisticRegression object that
        is applied to the natural cubic spline basis.  This
        is not used directly, but provided for reference.

    binary_splinecalibs: (multiclass) A list of the binary splinecalib objects
        used to do the multiclass calibration, indexed by class number.


    References
    ----------

    Lucena, B. Spline-Based Probability Calibration. https://arxiv.org/abs/1809.07751
   """
    def __init__(self, penalty='l2', solver='default',
                 knot_sample_size = 30, add_knots = None,
                 reg_param_vec = 'default', cv_spline=5, random_state=42,
                 unity_prior=True, unity_prior_gridsize=100, 
                 unity_prior_weight=100, max_iter=1000, tol=.0001,
                 logodds_scale=True, logodds_eps='auto', reg_prec=4,
                 force_knot_endpts=True):
        self.knot_sample_size = knot_sample_size
        self.add_knots = add_knots
        if (type(reg_param_vec)==str) and (reg_param_vec == 'default'):
            self.reg_param_vec = np.logspace(-4, 4, 17)
        else:
            self.reg_param_vec = np.array(reg_param_vec)
        self.cv_spline = cv_spline
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol
        self.penalty = penalty
        self.unity_prior = unity_prior
        self.unity_prior_weight = unity_prior_weight
        self.unity_prior_gridsize = unity_prior_gridsize
        self.reg_prec = reg_prec
        self.force_knot_endpts = force_knot_endpts
        self.logodds_scale = logodds_scale 
        if type(logodds_eps==str) and (logodds_eps=='auto'):
            self.logodds_eps_auto = True
            self.logodds_eps = .001
        else:
            self.logodds_eps = logodds_eps
            self.logodds_eps_auto=False
        self.solver = solver
        if (penalty=='l1') and not (solver in ['liblinear','saga']):
            if (solver!='default'):
                warnings.warn('L1 penalty requires "liblinear" or "saga" solver.\nUsing "liblinear"')
            self.solver = 'liblinear'
        if (penalty=='l2') and (solver=='default'):
            self.solver = 'lbfgs'

    def fit(self, y_model, y_true, verbose=False):
        """Fit the calibrator given a set of predictions and truth values.

        This method will fit the calibrator. It handles both binary and
        multiclass problems.

        Parameters
        ----------
        y_pred : array-like, shape (n_samples, n_classes)
            Model outputs on which to perform calibration.
            
            If passed a 1-d array of length (n_samples) this will be presumed
            to mean binary classification and the inputs presumed to be the 
            probability of class "1".

            If passed a 2-d array, it is assumed to be a multiclass calibration
            where the number of classes is n_classes.  Binary problems may
            take 1-d or 2-d arrays as y_pred.

        y_true : array-like, shape (n_samples)
            Truth values to calibrate against.  Values must be integers
            between 0 and n_classes-1
        """
        if len(y_model.shape)>1:
            # 2-d array
            self.n_classes =y_model.shape[1]
        else:
            self.n_classes = 2

        if self.n_classes > 2:
            self._fit_multiclass(y_model, y_true, verbose=verbose)
        else:
            if len(y_model.shape) == 2:
                y_model = y_model[:,1]
            self._fit_binary(y_model, y_true, verbose=verbose)

    def _fit_binary(self, y_model, y_true, verbose=False):
        
        if verbose:
            print("Determining Calibration Function")

        # Determine the knots to use (on probability scale)
        self.knot_vec = self._get_knot_vec(y_model)

        # Augment data with unity prior (if necessary)
        self.use_weights = False
        if self.unity_prior:
            scorevec_coda, truthvec_coda = create_yeqx_bias_vectors(self.unity_prior_gridsize)
            self.use_weights = True
            coda_wt = self.unity_prior_weight/((self.unity_prior_gridsize+1)*(self.unity_prior_gridsize))
            weightvec = np.concatenate((np.ones(len(y_model)), coda_wt * np.ones(len(scorevec_coda))))
            y_model = np.concatenate((y_model, scorevec_coda))
            y_true = np.concatenate((y_true, truthvec_coda))

        # map data and knots to log odds scale if necessary
        if self.logodds_scale:
            if self.logodds_eps_auto:
                self._compute_logodds_eps_from_data(y_model)

            self.knot_vec_tr = np.minimum(1-self.logodds_eps, self.knot_vec)
            self.knot_vec_tr = np.maximum(self.logodds_eps, self.knot_vec_tr)
            self.knot_vec_tr = np.unique(self.knot_vec_tr)
            self.knot_vec_tr = np.log(self.knot_vec_tr/(1-self.knot_vec_tr))

            y_model_tr = np.minimum(1-self.logodds_eps, y_model)
            y_model_tr = np.maximum(self.logodds_eps, y_model_tr)
            y_model_tr = np.log(y_model_tr/(1-y_model_tr))
        else:
            y_model_tr = y_model
            self.knot_vec_tr = self.knot_vec

        # compute basis expansion
        X_mat = _natural_cubic_spline_basis_expansion(y_model_tr, self.knot_vec_tr)

        # perform cross-validated logistic regression
        if self.use_weights:
            best_C, ll_vec, reg = logreg_cv(X_mat, y_true, self.cv_spline,
                                            self.reg_param_vec,
                                            weightvec=weightvec,
                                            penalty=self.penalty,
                                            solver=self.solver,
                                            max_iter=self.max_iter,
                                            tol=self.tol,
                                            reg_prec=self.reg_prec)
        else:
            best_C, ll_vec, reg = logreg_cv(X_mat, y_true, self.cv_spline,
                                            self.reg_param_vec,
                                            penalty=self.penalty,
                                            solver=self.solver,
                                            max_iter=self.max_iter,
                                            tol=self.tol,
                                            reg_prec=self.reg_prec)
        self.best_reg_param = best_C
        self.reg_param_scores = ll_vec
        self.basis_coef_vec = reg.coef_
        self.lrobj = reg

    def _fit_multiclass(self, y_model, y_true, verbose=False):
        self.binary_splinecalibs = []
        for i in range(self.n_classes):
            if verbose:
                print('Calibrating Class {}'.format(i))
            le_auto = 'auto' if self.logodds_eps_auto else self.logodds_eps
            self.binary_splinecalibs.append(SplineCalib(
                            penalty=self.penalty,
                            knot_sample_size = self.knot_sample_size,
                            add_knots = self.add_knots,
                            reg_param_vec = self.reg_param_vec,
                            cv_spline=self.cv_spline,
                            random_state=self.random_state,
                            max_iter=self.max_iter,
                            tol=self.tol,
                            solver=self.solver,
                            logodds_scale=self.logodds_scale,
                            logodds_eps=le_auto,
                            unity_prior = self.unity_prior,
                            unity_prior_weight = self.unity_prior_weight,
                            unity_prior_gridsize = self.unity_prior_gridsize))
            self.binary_splinecalibs[i].fit(y_model[:,i],
                                            (y_true==i).astype(int))

    def _get_knot_vec(self, y_model):
        """Routine to choose the set of knots."""
        random.seed(self.random_state)
        unique_vals = np.unique(y_model)
        num_unique = len(unique_vals)
        if(num_unique<3):
            raise Exception('Less than 3 unique input values.')

        if (self.knot_sample_size==0):
            if (self.add_knots is None):
                raise Exception('Must have knot_sample_size>0 or specify add_knots')
            else:
                return(np.unique(self.add_knots))

        if (self.knot_sample_size<2) and (self.force_knot_endpts):
            warn_msg = ('force_knot_endpts is True but knot_sample_size<2.\n' +
                       'Changing force_knot_endpts to False.')
            warnings.warn(warn_msg)
            self.force_knot_endpts=False

        # Usual case: more unique values than the sample being chosen
        if (num_unique > self.knot_sample_size):
            if self.force_knot_endpts:
                smallest_knot, biggest_knot = unique_vals[0], unique_vals[-1]
                other_vals = unique_vals[1:-1]
                random.shuffle(other_vals)
                curr_knot_vec = other_vals[:(self.knot_sample_size-2)]
                curr_knot_vec = np.concatenate((curr_knot_vec, 
                                                   [smallest_knot, biggest_knot]))
            else:
                random.shuffle(unique_vals)
                curr_knot_vec = unique_vals[:self.knot_sample_size]

        # use all the unique_vals
        else:
            curr_knot_vec = unique_vals

        # Add in the additional knots
        if self.add_knots is not None:
            add_knot_vec = np.array(self.add_knots)
            curr_knot_vec = np.concatenate((curr_knot_vec,add_knot_vec))
        
        # Sort and remove duplicates
        return(np.unique(curr_knot_vec))


    def calibrate(self, y_in):
        """Calibrates a set of predictions after being fit.

        This function returns calibrated probabilities after being
        fit on a set of predictions and their true answers.  It handles
        either binary and multiclass problems, depending on how it was fit.

        Parameters
        ----------
        y_in : array-like, shape (n_samples, n_features)
            The pre_calibrated scores.  For binary classification
            can pass in a 1-d array representing the probability
            of class 1.

        Returns
        -------
        y_out : array, shape (n_samples, n_classes)
            The calibrated probabilities: y_out will be returned
            in the same shape as y_in.
        """
        
        if self.n_classes>2:
            return(self._calibrate_multiclass(y_in))
        elif self.n_classes==2:
            return(self._calibrate_binary(y_in))
        else:
            warnings.warn('SplineCalib not fit or only one class found')

    def _calibrate_binary(self, y_in):
        if (len(y_in.shape)==2):
            if (y_in.shape[1]==2):
                y_in_to_use = y_in[:,1]
                two_col = True
            elif (y_in.shape[0]==1):
                y_in_to_use = y_in[:,0]
                two_col=False
        elif (len(y_in.shape)==1):
            y_in_to_use = y_in
            two_col = False
        else:
            warnings.warn('Unable to handle input of this shape')

        if self.logodds_scale:
            y_in_to_use = np.minimum(1-self.logodds_eps, y_in_to_use)
            y_in_to_use = np.maximum(self.logodds_eps, y_in_to_use)
            y_model_tr = np.log(y_in_to_use/(1-y_in_to_use))
        else:
            y_model_tr = y_in_to_use
        basis_exp = _natural_cubic_spline_basis_expansion(y_model_tr,self.knot_vec_tr)
        y_out = basis_exp.dot(self.basis_coef_vec.T)
        y_out = 1/(1+np.exp(-y_out))
        if (len(y_out.shape)>1):
            y_out = y_out[:,0]
        y_out = np.vstack((1-y_out, y_out)).T if two_col else y_out
        return y_out

    def _calibrate_multiclass(self, y_in):
        y_out = -1*np.ones(y_in.shape)
        for i in range(self.n_classes):
            y_out[:,i] = self.binary_splinecalibs[i].calibrate(y_in[:,i])
        y_out = (y_out.T/(np.sum(y_out, axis=1))).T
        return y_out

    def show_spline_reg_plot(self, class_num=None):
        """Plots the cross-val loss against the regularization parameter.

            This is a diagnostic tool, for example, to indicate whether or
            not the search over regularization parameter values was wide
            enough or dense enough.

            For multiclass calibration, the class number must be given."""
        if self.n_classes == 2:
            plt.plot(np.log10(self.reg_param_vec),self.reg_param_scores, marker='x')
            plt.xlabel('Regularization Parameter (log 10 scale)')
            plt.ylabel('Average Log Loss across folds')
            plt.axvline(np.log10(self.best_reg_param),0,1,color='black',linestyle='--')
        else:
            if class_num is None:
                warnings.warn('Must specify a class number for Multiclass Calibration')
            else:
                obj=self.binary_splinecalibs[class_num]
                plt.plot(np.log10(obj.reg_param_vec),obj.reg_param_scores, marker='x')
                plt.xlabel('Regularization Parameter (log 10 scale)')
                plt.ylabel('Average Log Loss across folds')
                plt.axvline(np.log10(obj.best_reg_param),0,1,color='black',linestyle='--')


    def show_calibration_curve(self, class_num=None, resolution=.001,
                               show_baseline=True, scaling='none',
                               scaling_base=10, scaling_eps=.0001):
        """Plots the calibration curve as a function from [0,1] to [0,1]

        Parameters
        ----------

        resolution: The space between the plotted points on the x-axis.

        show_baseline: Boolean - default=True.  Whether or not to show the line
            y=x as a reference.

        scaling: default is 'none'.  Alternative is 'logit' which scales the
            curve to show more detail close to 0 and 1.  When using in tandem
            with `plot_reliability_diagram` should be sure to use the same
            scaling parameters.

        scaling_eps: default is .0001.  Ignored unless scaling='logit'. This 
            indicates the smallest meaningful positive probability you
            want to consider.

        scaling_base: default is 10. Ignored unless scaling='logit'. This
            indicates the base used when scaling back and forth.  Matters
            only in how it affects the automatic tick marks.
        """
        tvec = np.linspace(0,1,int(np.ceil(1/resolution))+1)
        avec = np.logspace(-16,-3,6)
        bvec = 1-avec
        tvec = np.unique(np.concatenate((tvec,avec,bvec)))
        if self.n_classes == 2:
            if scaling=='none':
                plt.plot(tvec, self.calibrate(tvec))
                if show_baseline:
                    plt.plot(tvec,tvec,'k--')
                plt.axis([-0.1,1.1,-0.1,1.1])
            elif scaling=='logit':
                tvec_to_plot = my_logit(tvec, base=scaling_base)
                y_to_plot = my_logit(self.calibrate(tvec), base=scaling_base)
                plt.plot(tvec_to_plot, y_to_plot)
                if show_baseline:
                    plt.plot(tvec_to_plot,tvec_to_plot,'k--')

            plt.xlabel('Uncalibrated')
            plt.ylabel('Calibrated')

        elif class_num is None:
                warnings.warn('Must specify a class number for Multiclass Calibration')
        else:
            self.binary_splinecalibs[class_num].show_calibration_curve(
                                                    resolution=resolution,
                                                    show_baseline=show_baseline,
                                                    scaling=scaling,
                                                    scaling_base=scaling_base,
                                                    scaling_eps=scaling_eps)

    def transform(self, y_in):
        """alias for calibrate"""
        return self.calibrate(y_in)

    def predict(self, y_in):
        """alias for calibrate"""
        return self.calibrate(y_in)

    def predict_proba(self, y_in):
        """alias for calibrate"""
        return self.calibrate(y_in)

    def _compute_logodds_eps_from_data(self, y_model):
        """Routine to automatically choose the logodds_eps value"""
        y_model_loe = y_model[(y_model>0) & (y_model<1)]
        closest_to_zero = np.min(y_model_loe)
        closest_to_one = 1-np.min(1-y_model_loe)
        loe_exp = np.round(np.log10(np.min((closest_to_zero,
                                            closest_to_one))))-1
        self.logodds_eps = np.minimum(self.logodds_eps, 10**loe_exp)
