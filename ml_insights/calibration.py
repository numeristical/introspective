"""Calibration of predicted probabilities."""
import numpy as np
import sklearn
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin, clone

try:
    from sklearn.model_selection import StratifiedKFold
except:
    from sklearn.cross_validation import StratifiedKFold

from .calibration_utils import prob_calibration_function, compact_logit


class SplineCalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    """Probability calibration using cubic splines.

    With this class, the base_estimator is fit on each of the cross-validation
    training set folds in order to generate scores on the (cross-validated)
    test set folds.  The test set scores are accumulated into a final vector
    (the size of the full set) which is used to calibrate the answers.
    The model is then fit on the full data set.  The predict, and predict_proba
    methods are then updated to use the combination of the predictions from the 
    full model and the calibration function computed as above.

    Parameters
    ----------
    base_estimator : instance BaseEstimator
        The classifier whose output decision function needs to be calibrated
        to offer more accurate predict_proba outputs. If cv='prefit', the
        classifier must have been fit already on data.

    method : 'logistic' or 'ridge'
        The default is 'logistic', which is best if you plan to use log-loss as your
        performance metric.  This method is relatively robust and will typically do
        well on brier score as well.  The 'ridge' method calibrates using an L2 loss,
        and therefore should do better for brier score, but may do considerably worse
        on log-loss.

    cv : integer, cross-validation generator, iterable or "prefit", optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - 'prefit', if you wish to use the data only for calibration

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If ``y`` is
        neither binary nor multiclass, :class:`sklearn.model_selection.KFold`
        is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        If "prefit" is passed, it is assumed that base_estimator has been
        fitted already and all data is used for calibration.

    Attributes
    ----------
    uncalibrated_classifier: this gives the uncalibrated version of the classifier, fit on the entire data set

    calib_func: this is the calibration function that has been learned from the cross-validation.  Applying this function
     to the results of the uncalibrated classifier (via model.predict_proba(X_test)[:,1]) gives the fully calibrated classifier

    References
    ----------
   """
    def __init__(self, base_estimator=None, method='logistic', cv=5, transform_type='none', cl_eps = .000001, **calib_kwargs):
        warn_msg = ('\nThis class is deprecated and will eventually be removed.' + 
                    '\nPlease use the SplineCalib class for calibration.')
        warnings.warn(warn_msg, FutureWarning)

        self.base_estimator = base_estimator
        self.uncalibrated_classifier = None
        self.calib_func = None
        self.method = method
        self.cv = cv
        self.cl_eps = cl_eps
        self.calib_kwargs = calib_kwargs
        self.fit_on_multiclass = False
        self.transform_type = transform_type
        self.pre_transform = lambda x: x
        if type(self.transform_type) == str:
            if self.transform_type == 'cl':
                self.pre_transform = lambda x: compact_logit(x, eps = self.cl_eps)
        if callable(self.transform_type):
            self.pre_transform = self.transform_type

    def fit(self, X, y, verbose=False):
        """Fit the calibrated model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        
        if len(np.unique(y)) > 2:
            self.fit_on_multiclass = True
            return self._fit_multiclass(X, y, verbose=verbose)

        self.fit_on_multiclass=False
        if ((type(self.cv)==str) and (self.cv=='prefit')):
            self.uncalibrated_classifier = self.base_estimator
            y_pred = self.uncalibrated_classifier.predict_proba(X)[:,1]

        else:
            y_pred = np.zeros(len(y))
            
            if sklearn.__version__ < '0.18':
                if type(self.cv)==int:
                    skf = StratifiedKFold(y, n_folds=self.cv,shuffle=True)
                else:
                    skf = self.cv
            else:
                if type(self.cv)==int:
                    skf = StratifiedKFold(n_splits=self.cv, shuffle=True).split(X, y)
                else:
                    skf = self.cv.split(X,y)
            for idx, (train_idx, test_idx) in enumerate(skf):
                if verbose:
                    print("training fold {} of {}".format(idx+1, self.cv))
                X_train = np.array(X)[train_idx,:]
                X_test = np.array(X)[test_idx,:]
                y_train = np.array(y)[train_idx]
                # We could also copy the model first and then fit it
                this_estimator = clone(self.base_estimator)
                this_estimator.fit(X_train,y_train)
                y_pred[test_idx] = this_estimator.predict_proba(X_test)[:,1]
            
            if verbose:
                print("Training Full Model")
            self.uncalibrated_classifier = clone(self.base_estimator)
            self.uncalibrated_classifier.fit(X, y)

        # calibrating function
        if verbose:
            print("Determining Calibration Function")
        if self.method=='logistic':
            self.calib_func = prob_calibration_function(y, self.pre_transform(y_pred), verbose=verbose, **self.calib_kwargs)
        if self.method=='ridge':
            self.calib_func = prob_calibration_function(y, self.pre_transform(y_pred), method='ridge', verbose=verbose, **self.calib_kwargs)
        # training full model

        return self

    def _fit_multiclass(self, X, y, verbose=False):
        """Fit the calibrated model in multiclass setting

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        class_list = np.unique(y)
        num_classes = len(class_list)
        y_mod = np.zeros(len(y))
        for i in range(num_classes):
           y_mod[y==class_list[i]]=i

        y_mod = y_mod.astype(int)
        if ((type(self.cv)==str) and (self.cv=='prefit')):
            self.uncalibrated_classifier = self.base_estimator
            y_pred = self.uncalibrated_classifier.predict_proba(X)

        else:
            y_pred = np.zeros((len(y_mod),num_classes))
            if sklearn.__version__ < '0.18':
                skf = StratifiedKFold(y_mod, n_folds=self.cv,shuffle=True)
            else:
                skf = StratifiedKFold(n_splits=self.cv, shuffle=True).split(X, y)
            for idx, (train_idx, test_idx) in enumerate(skf):
                if verbose:
                    print("training fold {} of {}".format(idx+1, self.cv))
                X_train = np.array(X)[train_idx,:]
                X_test = np.array(X)[test_idx,:]
                y_train = np.array(y_mod)[train_idx]
                # We could also copy the model first and then fit it
                this_estimator = clone(self.base_estimator)
                this_estimator.fit(X_train,y_train)
                y_pred[test_idx,:] = this_estimator.predict_proba(X_test)
            
            if verbose:
                print("Training Full Model")
            self.uncalibrated_classifier = clone(self.base_estimator)
            self.uncalibrated_classifier.fit(X, y_mod)

        # calibrating function
        if verbose:
            print("Determining Calibration Function")
        if self.method=='logistic':
            self.calib_func, self.cf_list = prob_calibration_function_multiclass(y_mod, self.pre_transform(y_pred), verbose=verbose, **self.calib_kwargs)
        if self.method=='ridge':
            self.calib_func, self.cf_list = prob_calibration_function_multiclass(y_mod, self.pre_transform(y_pred), verbose=verbose, method='ridge', **self.calib_kwargs)
        # training full model

        return self


    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas.
        """
        # check_is_fitted(self, ["classes_", "calibrated_classifier"])
        if self.fit_on_multiclass:
            return self.calib_func(self.pre_transform(self.uncalibrated_classifier.predict_proba(X)))
        
        col_1 = self.calib_func(self.pre_transform(self.uncalibrated_classifier.predict_proba(X)[:,1]))
        col_0 = 1-col_1
        return np.vstack((col_0,col_1)).T
        
            

    def predict(self, X):
        """Predict the target of new samples. Can be different from the
        prediction of the uncalibrated classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples,)
            The predicted class.
        """
        # check_is_fitted(self, ["classes_", "calibrated_classifier"])
        return self.uncalibrated_classifier.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def classes_(self):
        return self.uncalibrated_classifier.classes_



"""Calibration of predicted probabilities."""
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone

try:
    from sklearn.model_selection import StratifiedKFold
except:
    from sklearn.cross_validation import StratifiedKFold

from .calibration_utils import prob_calibration_function_multiclass


class SplineCalibratedClassifierMulticlassCV(BaseEstimator, ClassifierMixin):
    """Probability calibration using cubic splines.

    With this class, the base_estimator is fit on each of the cross-validation
    training set folds in order to generate scores on the (cross-validated)
    test set folds.  The test set scores are accumulated into a final vector
    (the size of the full set) which is used to calibrate the answers.
    The model is then fit on the full data set.  The predict, and predict_proba
    methods are then updated to use the combination of the predictions from the 
    full model and the calibration function computed as above.

    Parameters
    ----------
    base_estimator : instance BaseEstimator
        The classifier whose output decision function needs to be calibrated
        to offer more accurate predict_proba outputs. If cv='prefit', the
        classifier must have been fit already on data.

    method : 'logistic' or 'ridge'
        The default is 'logistic', which is best if you plan to use log-loss as your
        performance metric.  This method is relatively robust and will typically do
        well on brier score as well.  The 'ridge' method calibrates using an L2 loss,
        and therefore should do better for brier score, but may do considerably worse
        on log-loss.

    cv : integer, cross-validation generator, iterable or "prefit", optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - 'prefit', if you wish to use the data only for calibration

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. If ``y`` is
        neither binary nor multiclass, :class:`sklearn.model_selection.KFold`
        is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        If "prefit" is passed, it is assumed that base_estimator has been
        fitted already and all data is used for calibration.

    Attributes
    ----------
    uncalibrated_classifier: this gives the uncalibrated version of the classifier, fit on the entire data set

    calib_func: this is the calibration function that has been learned from the cross-validation.  Applying this function
     to the results of the uncalibrated classifier (via model.predict_proba(X_test)[:,1]) gives the fully calibrated classifier

    References
    ----------
   """
    def __init__(self, base_estimator=None, method='logistic', cv=5, **calib_kwargs):
        warn_msg = ('\nThis class is deprecated and will eventually be removed.' + 
                    '\nPlease use the SplineCalib class for calibration.')
        warnings.warn(warn_msg, FutureWarning)

        self.base_estimator = base_estimator
        self.uncalibrated_classifier = None
        self.calib_func = None
        self.method = method
        self.cv = cv
        self.calib_kwargs = calib_kwargs

    def fit(self, X, y, verbose=False):
        """Fit the calibrated model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        class_list = np.unique(y)
        num_classes = len(class_list)
        y_mod = np.zeros(len(y))

        for i in range(num_classes):
            y_mod[np.where(y==class_list[i])]=i

        y_mod = y_mod.astype(int)
        if ((type(self.cv)==str) and (self.cv=='prefit')):
            self.uncalibrated_classifier = self.base_estimator
            y_pred = self.uncalibrated_classifier.predict_proba(X)[:,1]

        else:
            y_pred = np.zeros((len(y_mod),num_classes))
            if sklearn.__version__ < '0.18':
                skf = StratifiedKFold(y_mod, n_folds=self.cv,shuffle=True)
            else:
                skf = StratifiedKFold(n_splits=self.cv, shuffle=True).split(X, y)
            for idx, (train_idx, test_idx) in enumerate(skf):
                if verbose:
                    print("training fold {} of {}".format(idx+1, self.cv))
                X_train = np.array(X)[train_idx,:]
                X_test = np.array(X)[test_idx,:]
                y_train = np.array(y_mod)[train_idx]
                # We could also copy the model first and then fit it
                this_estimator = clone(self.base_estimator)
                this_estimator.fit(X_train,y_train)
                y_pred[test_idx,:] = this_estimator.predict_proba(X_test)
            
            if verbose:
                print("Training Full Model")
            self.uncalibrated_classifier = clone(self.base_estimator)
            self.uncalibrated_classifier.fit(X, y_mod)

        # calibrating function
        if verbose:
            print("Determining Calibration Function")
        if self.method=='logistic':
            self.calib_func = prob_calibration_function_multiclass(y_mod, y_pred, verbose=verbose, **self.calib_kwargs)
        if self.method=='ridge':
            self.calib_func = prob_calibration_function_multiclass(y_mod, y_pred, verbose=verbose, method='ridge', **self.calib_kwargs)
        # training full model

        return self

    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas.
        """
        # check_is_fitted(self, ["classes_", "calibrated_classifier"])
        return self.calib_func(self.uncalibrated_classifier.predict_proba(X))


    def predict(self, X):
        """Predict the target of new samples. Can be different from the
        prediction of the uncalibrated classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples,)
            The predicted class.
        """
        # check_is_fitted(self, ["classes_", "calibrated_classifier"])
        return self.uncalibrated_classifier.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def classes_(self):
        return self.uncalibrated_classifier.classes_
