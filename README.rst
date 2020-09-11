ML Insights
===========

Welcome to ML-Insights!

This package contains two core sets of functions:

1) Calibration
2) Interpreting Models

For probability calibration, the main class is `SplineCalib`.  Given a set of model outputs and the "true" classes, you can `fit` a SplineCalib object.  That object can then be used to `calibrate` future model predictions post-hoc.

.. code-block:: python

    >>> model.fit(X_train, y_train)
    >>> sc = mli.SplineCalib()
    >>> sc.fit(X_valid, y_valid)
    >>> uncalib_preds = model.predict_proba(X_test)
    >>> calib_preds = sc.calibrate(uncalib_preds)


.. code-block:: python

    >>> cv_preds = mli.cv_predictions(model, X_train, y_train)
    >>> model.fit(X_train, y_train)
    >>> sc = mli.SplineCalib()
    >>> sc.fit(cv_preds, y_train)
    >>> uncalib_preds = model.predict_proba(X_test)
    >>> calib_preds = sc.calibrate(uncalib_preds)



For model interpretability, we provide the `ice_plot` and `histogram_pair` functions as well as other tools.


.. code-block:: python

    >>> rd = mli.get_range_dict(X_train)
    >>> mli.ice_plot(model, X_test.sample(3), X_train.columns, rd)

.. code-block:: python

    >>> mli.histogram_pair(df.outcome, df.feature, bins=np.linspace(0,100,11))

Please see the documentation and examples at the links below.


- `Documentation <https://ml-insights.readthedocs.io>`_
- `Notebook Examples and Usage <https://github.com/numeristical/introspective/tree/master/examples>`_


Python
------
Python 3.4+


Disclaimer
==========

We have tested this tool to the best of our ability, but understand that it may have bugs.  It was most recently developed on Python 3.7.3.  Use at your own risk, but feel free to report any bugs to our github. <https://github.com/numeristical/introspective>


Installation
=============

.. code-block:: bash

    $ pip install ml_insights


Usage
======

.. code-block:: python

    >>> import ml_insights as mli
    >>> xray = mli.ModelXRay(model, data)

.. code-block:: python

	>>> rfm = RandomForestClassifier(n_estimators = 500, class_weight='balanced_subsample')
	>>> rfm_cv = mli.SplineCalibratedClassifierCV(rfm)
	>>> rfm_cv.fit(X_train,y_train)
	>>> test_res_calib_cv = rfm_cv.predict_proba(X_test)[:,1]
	>>> log_loss(y_test,test_res_calib_cv)

Source
======

Find the latest version on github: https://github.com/numeristical/introspective

Feel free to fork and contribute!

License
=======

Free software: `MIT license <LICENSE>`_

Developed By
============

- Brian Lucena
- Ramesh Sampath

References
==========

Alex Goldstein, Adam Kapelner, Justin Bleich, and Emil Pitkin. 2014. Peeking Inside the Black Box: Visualizing Statistical Learning With Plots of Individual Conditional Expectation. Journal of Computational and Graphical Statistics (March 2014)