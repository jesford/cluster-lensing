Tutorial
========

A complete walkthrough of all the functionality and tools available in this project
is provided in this Jupyter Notebook: `demo.ipynb <https://github.com/jesford/cluster-lensing/blob/master/demo.ipynb>`_.

A simple example use case is as follows. Suppose you want the
differential surface mass density :math:`\Delta\Sigma(r)` profiles for
a handful of galaxy clusters. Lets say they are at redshifts :math:`z
= 0.1`, :math:`0.2`, and :math:`0.5`, and have masses of :math:`1
\times 10^{15}`,  :math:`5 \times 10^{14}`, and :math:`2 \times
10^{14} M_{\odot}`, respectively.

After installing the :py:obj:`cluster-lensing` package, all we have to do is:

.. code-block:: python
		
		import numpy as np
		from cluster_lensing import clusters
		z = [0.1, 0.2, 0.3]
		c = clusters.ClusterEnsemble(z)
		c.m200 = [1e15, 5e14, 2e14]
		r = np.arange(0.5, 5, 0.5)  #radial bins
		c.calc_nfw(r)

Then the attribute :py:obj:`c.deltasigma_nfw` will contain an array of
:math:`\Delta\Sigma(r)` profiles, one for each of the three clusters:
      
>>> print c.deltasigma_nfw
[[ 216.99031097  131.96892957   89.95900137   65.95785776   50.81725977
    40.57785901   33.28891018   27.89244619   23.77114581]
 [ 159.82908955   88.92279328   57.75958551   41.06296211   30.95764522
    24.32100583   19.69970451   16.33693743   13.80449899]
 [  99.4563379    49.5200608    30.40868664   20.87864071   15.36566757
    11.85760144    9.47172553    7.76726675    6.50260522]] solMass / pc2


Let's say you are concerned about the accuracy of your clusters'
centroid estimates. We can easily calculate the miscentered
:math:`\Delta\Sigma(r)` profiles by passing the optional offsets
parameter to the :py:obj:`calc_nfw()` function. The offsets are given in units
of Mpc, just like the radius, and represent the width of a 2D Gaussian
offset distribution.

>>> c.calc_nfw(r, offsets=[0.1, 0.1, 0.1])
>>> print c.deltasigma_nfw
[[ 198.81572771  129.96652087   89.16550619   65.7334123    50.65140495
    40.49991719   33.22670747   27.85454194   23.73961746]
 [ 146.78755133   88.37783273   57.34742622   40.97849669   30.88520548
    24.29289523   19.67582995   16.32416666   13.79347989]
 [  91.77806816   49.70092894   30.26422875   20.87331318   15.3514408
    11.85804354    9.47045213    7.76873388    6.50334115]] solMass / pc2

The above example illustrates a really simple use case, but
:py:obj:`cluster-lensing` provides far more functionality. You can specify
cosmologies and mass-concentration relations to use for the NFW
calculations. Additionally, if you don't have mass estimates at all,
you can empoy a built-in (or customizeable) mass-richness scaling
relation to easily convert richness estimates into masses. Be sure to
check out the `Jupyter Notebook demonstration <https://github.com/jesford/cluster-lensing/blob/master/demo.ipynb>`_ of these tools.
