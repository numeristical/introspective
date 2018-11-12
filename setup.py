#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    # TODO: put package requirements here
    "pandas",
    "numpy"
]

setup(
    name='ml_insights',
    version='0.0.18',
    description="Package to understand ML Models",
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
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ]
)
