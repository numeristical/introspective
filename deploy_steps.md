To Develop Locally:

1. Run `pip install -e .`


To Deploy to Test Server:

1. Run `python setup.py sdist bdist_wheel`

2. Upload using twine - `twine upload -r test dist/ml_insights-0.0.<version>*`
3. Install from Test PyPi - `pip install -i https://testpypi.python.org/pypi ml_insights --upgrade`

To Deploy to PyPi Server:

1. Run `python setup.py sdist bdist_wheel`

2. Upload using twine - `twine upload dist/ml_insights-0.0.<version>*`
3. Install from Test PyPi - `pip install ml_insights --upgrade`


**Note: When uploading to TestPyPi or PyPi, we cannot update the versions.  Version numbers need to be updated in setup.py and ml_insights/__init__.py


To Upgrade Documentation:

