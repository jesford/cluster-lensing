import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

DESCRIPTION = "Galaxy Cluster and Weak Lensing Tools"
LONG_DESCRIPTION = """
cluster-lensing: galaxy cluster halo calculations
======================================================
This package includes tools for calculating a variety of galaxy cluster properties, as well as mass-richness and mass-concentration scaling relations, and weak lensing profiles. These include surface mass density (Sigma) and differential surface mass density (DeltaSigma) for NFW halos, both with and without the effects of cluster miscentering.
For more information, visit http://jesford.github.io/cluster-lensing
"""
NAME = "cluster-lensing"
AUTHOR = "Jes Ford"
AUTHOR_EMAIL = "jesford@uw.edu"
MAINTAINER = "Jes Ford"
MAINTAINER_EMAIL = "jesford@uw.edu"
URL = 'http://github.com/jesford/cluster-lensing'
DOWNLOAD_URL = 'http://github.com/jesford/cluster-lensing'
LICENSE = 'MIT'
VERSION = '0.1.2'

setup(name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    packages=['clusterlensing', 'clusterlensing/tests'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'],
     )
