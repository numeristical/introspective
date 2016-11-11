.. ML Insights documentation master file, created by
   sphinx-quickstart on Wed Nov  9 13:32:07 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ML Insights's documentation!
=======================================

Contents:

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

Examples:
---------

`Use cases and How-To Examples
<https://github.com/numeristical/introspective/tree/master/examples>`_.


API Docs:
---------

.. autoclass:: ml_insights.ModelXRay
   :members:
