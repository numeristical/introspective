#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    "pandas>=0.23",
    "numpy>=1.16.0",
    "matplotlib>=2.0.0",
    "scikit-learn>=0.24.2",
    "scipy>=1.6.0",
    "splinecalib>=0.0.2"
]

setup(
    name='ml_insights',
    version='1.0.2',
    description="Package to calibrate and understand ML Models",
    long_description=readme,
    author="Brian Lucena / Ramesh Sampath",
    author_email='brianlucena@gmail.com',
    url='http://ml-insights.readthedocs.io/en/latest/',
    packages=[
        'ml_insights',
    ],
    package_dir={'ml_insights':
                 'ml_insights'},
    include_package_data=True,
    install_requires=requirements,

    license="MIT license",
    zip_safe=False,
    keywords='ml_insights',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ]
)
