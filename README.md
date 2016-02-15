## Tools for fitting Weak Lensing profiles

A Python repository in progress for modules that are helpful in
fitting weak lensing shear or magnification profiles.

Currently, the focus of this repository is on the ClusterEnsemble()
class in clusters.py. See a demo of what it can do in the provided
notebook: [demo.ipynb](https://github.com/jesford/wl-profile/blob/master/demo.ipynb).

ClusterEnsemble() allows you to easily build up a nicely
formatted table (a pandas dataframe) of cluster attributes, and
automatically generates parameters that depend on each other. It uses
a customizable powerlaw mass-richness scaling relation to convert richness
N<sub>200</sub> to mass M<sub>200</sub>, and to generate other parameters.

The calc_nfw() method calculates the NFW profiles for Sigma(r) and
DeltaSigma(r), which are useful for fitting weak lensing shear or
magnification profiles. Optionally, it will calculate the **miscentered**
profiles, given an offset parameter describing the width of the
Gaussian miscentering offset distribution. See, for example, 
[Ford et al. 2015](http://arxiv.org/abs/1409.3571), for the
miscentering formalism, and an example use case. All of the code you
see in this repository (as well as the repositories linked below) is a
cleaned up version of the same code used for that
[CFHTLenS cluster shear paper](http://arxiv.org/abs/1409.3571), as
well as for our previous [cluster magnification paper](http://arxiv.org/abs/1310.2295).

Running the ClusterEnsemble() module requires you to additionally
download the code for doing
[concentration-mass relationships](https://github.com/jesford/cofm)
and for calculating the
[NFW profiles](https://github.com/jesford/smd-nfw). The
latter of these is currently being entirely revamped (it was
previously only implemented in C, has recently been translated to
Python, but still has some issues to work out). Soon the scattered
bits and pieces will be collected into an organized package, so check
back later for updates.
