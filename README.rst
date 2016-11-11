ML Insights
===========

Package to understand Supervised ML Models.  This package has been tested with Scikit-Learn and XGBoost library.  It should work with any machine learning library that has a `predict` and `predict_proba` methods for regression and classification estimators.

The main entry point to this package is `ModelXRay` class.  Instantiate it with the model and data.  The data can be what the model was trained with, but inteded to be used for out of bag or test data to see how the model performs when one feature is changed, holding everything else constant.

- `API Docs <https://ml-insights.readthedocs.io>`_
- `Notebook Examples and Usage <https://github.com/numeristical/introspective/tree/master/examples>`_

We have not tested this for unsupervied models.

Python
------
Python 2.7 and 3.4+


Disclaimer
==========

We have tested this tool to the best of my ability, but understand that it may have bugs. Use at your own risk!


Installation
=============

.. code-block:: bash

    $ pip install ml_insights


Usage
======

.. code-block:: python

    >>> import ml_insights as mli
    >>> xray = mli.ModelXRay(model, data)


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
