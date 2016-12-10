"""Calibration of predicted probabilities."""
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone

try:
    from sklearn.model_selection import StratifiedKFold
except:
    from sklearn.cross_validation import StratifiedKFold

from .calibration_utils import prob_calibration_function


class CalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    """Probability calibration with isotonic regression or sigmoid.

    With this class, the base_estimator is fit on the train set of the
    cross-validation generator and the test set is used for calibration.
    The probabilities for each of the folds are then averaged
    for prediction. In case that cv="prefit" is passed to __init__,
    it is assumed that base_estimator has been fitted already and all
    data is used for calibration. Note that data for fitting the
    classifier and for calibrating it must be disjoint.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    base_estimator : instance BaseEstimator
        The classifier whose output decision function needs to be calibrated
        to offer more accurate predict_proba outputs. If cv=prefit, the
        classifier must have been fit already on data.

    method : 'sigmoid' or 'isotonic'
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method or 'isotonic' which is a
        non-parametric approach. It is not advised to use isotonic calibration
        with too few calibration samples ``(<<1000)`` since it tends to
        overfit.
        Use sigmoids (Platt's calibration) in this case.

    cv : integer, cross-validation generator, iterable or "prefit", optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

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
    classes_ : array, shape (n_classes)
        The class labels.

    calibrated_classifiers_: list (len() equal to cv or 1 if cv == "prefit")
        The list of calibrated classifiers, one for each crossvalidation fold,
        which has been fitted on all but the validation fold and calibrated
        on the validation fold.

    References
    ----------
    .. [1] Obtaining calibrated probability estimates from decision trees
           and naive Bayesian classifiers, B. Zadrozny & C. Elkan, ICML 2001

    .. [2] Transforming Classifier Scores into Accurate Multiclass
           Probability Estimates, B. Zadrozny & C. Elkan, (KDD 2002)

    .. [3] Probabilistic Outputs for Support Vector Machines and Comparisons to
           Regularized Likelihood Methods, J. Platt, (1999)

    .. [4] Predicting Good Probabilities with Supervised Learning,
           A. Niculescu-Mizil & R. Caruana, ICML 2005
    """
    def __init__(self, base_estimator=None, method='sigmoid', cv=5, **calib_kwargs):
        self.base_estimator = base_estimator
        self.calibrated_classifier = None
        self.method = method
        self.cv = cv
        self.calib_kwargs = calib_kwargs

    def fit(self, X, y):
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
        y_pred = np.zeros(len(y))
        if sklearn.__version__ < '0.18':
            skf = StratifiedKFold(y, n_folds=self.cv,shuffle=True)
        else:
            skf = StratifiedKFold(n_splits=self.cv, shuffle=True).split(X, y)
        for idx, (train_idx, test_idx) in enumerate(skf):
            print("training fold {} of {}".format(idx+1, self.cv))
            X_train = np.array(X)[train_idx,:]
            X_test = np.array(X)[test_idx,:]
            y_train = np.array(y)[train_idx]
            # We could also copy the model first and then fit it
            this_estimator = clone(self.base_estimator)
            this_estimator.fit(X_train,y_train)
            y_pred[test_idx] = this_estimator.predict_proba(X_test)[:,1]

        # calibrating function
        self.calib_func = prob_calibration_function(y, y_pred, **self.calib_kwargs)
        # training full model
        print("Training Full Model")
        self.calibrated_classifier = clone(self.base_estimator)
        self.calibrated_classifier.fit(X, y)

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
        col_1 = self.calib_func(self.calibrated_classifier.predict_proba(X)[:,1])
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
        return self.calibrated_classifier.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def classes_(self):
        return self.calibrated_classifier.classes_
