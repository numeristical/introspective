.. ML Insights documentation master file, created by
   sphinx-quickstart on Wed Nov  9 13:32:07 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ML Insights's documentation!
=======================================

Contents:

This package currently contains two useful sets of features.  The first is around the Model X-ray, which gives
some ways to understand black-box models.  The second is around probability calibration.

.. toctree::
   :maxdepth: 2

Installation:
-------------

.. code-block:: bash

    $ pip install ml_insights


Usage:
------

.. code-block:: python

    >>> import ml_insights as mli
    >>> xray = mli.ModelXRay(model, data)

.. code-block:: python

	>>> rfm = RandomForestClassifier(n_estimators = 500, class_weight='balanced_subsample')
	>>> rfm_cv = mli.SplineCalibratedClassifierCV(rfm)
	>>> rfm_cv.fit(X_train,y_train)
	>>> test_res_calib_cv = rfm_cv.predict_proba(X_test)[:,1]
	>>> log_loss(y_test,test_res_calib_cv)


Examples:
---------

`Use cases and How-To Examples
<https://github.com/numeristical/introspective/tree/master/examples>`_.


API Docs:
---------

.. autoclass:: ml_insights.ModelXRay
   :members:
