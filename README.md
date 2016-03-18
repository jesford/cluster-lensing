## Galaxy Cluster and Weak Lensing Tools
[![Build Status](https://travis-ci.org/jesford/cluster-lensing.svg?branch=master)](https://travis-ci.org/jesford/cluster-lensing)

The full **cluster-lensing** documentation is [here](http://jesford.github.io/cluster-lensing).

###Installation
**cluster-lensing** is a pure Python package that can be installed by
running:
```
$ pip install cluster-lensing
```
This package runs on Python 2.7, 3.4, and 3.5, and its dependencies include numpy,
scipy, astropy, and pandas.

### Description
**cluster-lensing** is a Python project for calculating a variety of galaxy cluster
properties, as well as mass-richness and mass-concentration scaling
relations, and weak lensing profiles. These include surface mass
density (Sigma) and differential surface mass density (DeltaSigma) for
NFW halos, both with and without the effects of cluster miscentering.

The focus of this project is the ClusterEnsemble()
class in clusters.py. See a demo of what it can do in the provided
notebook: [demo.ipynb](https://github.com/jesford/cluster-lensing/blob/master/demo.ipynb).

ClusterEnsemble() allows you to easily build up a nicely
formatted table (a pandas dataframe) of cluster attributes, and
automatically generates parameters that depend on each other. It uses
a customizable powerlaw mass-richness scaling relation to convert
between richness N<sub>200</sub> and mass M<sub>200</sub>, and to
generate other parameters. Other customizeable options include
specifications of the cosmology and a choice of several
concentration-mass relationships from the literature.

The ClusterEnsemble.calc_nfw() method provides simplified access to the
SurfaceMassDensity() class in nfw.py. The latter calculates the NFW halo
profiles for Sigma(r) and DeltaSigma(r), which are useful for fitting
weak lensing shear or magnification profiles. Optionally, it will
calculate the **miscentered** profiles, given an offset parameter
describing the width of the 2D Gaussian miscentering offset
distribution. See, for example, 
[Ford et al. 2015](http://arxiv.org/abs/1409.3571), for the
miscentering formalism, and an example use case. All of the code you
see in this repository (as well as the repositories linked below) is a
cleaned up version of the same code used for that
[CFHTLenS cluster shear paper](http://arxiv.org/abs/1409.3571), as
well as for our previous [cluster magnification paper](http://arxiv.org/abs/1310.2295).

This project has inherited code from the
[cofm](https://github.com/jesford/cofm) repository for
concentration-mass relationships and the
[smd-nfw](https://github.com/jesford/smd-nfw) repository for
calculating NFW halo profiles.
