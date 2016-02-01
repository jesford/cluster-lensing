"""Galaxy Cluster Ensemble Calculations.

Cluster mass-richness and mass-concentration scaling relations, and NFW
halo profiles for weak lensing shear and magnification, including the 
effects of cluster miscentering offsets.

This framework calculates properties and profiles for every individual
cluster, storing the data in tabular form. This is useful for fitting
measured stacked weak lensing profiles, e.g. when you want to account for
the full redshift, mass, and/or centroid offset distributions, and avoid
fitting a single average mass at a single effective redshift.
"""

#Useful documentation site:
#https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from astropy.cosmology import Planck13 as cosmo
from astropy import units
import os

from cofm import c_DuttonMaccio as calc_c200
from smd_nfw import SurfaceMassDensity
import utils

try:
    from IPython.display import display
    notebook_display = True
except:
    notebook_display = False


class ClusterEnsemble(object):
    """Ensemble of galaxy clusters and their properties.

    The ClusterEnsemble object contains parameters and calculated
    values for every individual cluster in a sample. Initializing with
    a collection of redshifts z will fix the number of clusters
    described by the object. Setting the n200 values will then populate
    the object with the full set of available attributes (except for
    the NFW profiles). The values of z, n200, massrich_norm, and
    massrich_slope can be altered and these changes will propogate to
    the other (dependent) attributes.

    In order to generate the attributes containing the NFW halo
    profiles, sigma_nfw and deltasigma_nfw, pass the desired radial
    bins (and additional optional parameters) to the calc_nfw method.

    Parameters
    ----------
    z : array_like
        Redshifts for each cluster in the sample. Should be 1D.

    Attributes
    ----------
    z : ndarray
        Cluster redshifts. Setter function available.
    n200 : ndarray
        Cluster richness. Setter function available.
    m200 : Quantity
        Cluster masses (1D ndarray) with astropy.units of Msun.
    c200 : ndarray
        Cluster concentration parameters.
    delta_c : ndarray
        Characteristic overdensities of the cluster halos.
    r200 : Quantity
        Cluster radii (1D ndarray) with astropy.units of Mpc.
    rs : Quantity
        Cluster scale radii (1D ndarray) with astropy.units of Mpc.
    Dang_l : Quantity
        Angular diameter distances from z=0 to z, (1D ndarray) with
        astropy.units of Mpc.
    dataframe : DataFrame
        Cluster information in tabular form (Pandas DataFrame).
    massrich_norm : Quantity
        Normalization of mass-richness relation, (scalar) with
        astropy.units of Msun. Default value is 2.7e+13 Msun. Setter
        function available.
    massrich_slope : float
        Slope of mass-richness relation. Default value is 1.4. Setter
        function available.
    describe : str
        Short description of the ClusterEnsemble object.
    number : int
        Number of clusters in the sample.
    sigma_nfw : Quantity
        Surface mass density of every cluster, (1D ndarray) with
        astropy.units of Msun/pc/pc. Generated for each element of rbins
        by running calc_nfw(rbins) method.
    deltasigma_nfw : Quantity
        Differential surface mass density of every cluster, (1D ndarray)
        with astropy.units of Msun/pc/pc. Generated for each element of
        rbins by running calc_nfw(rbins) method.

    Methods
    ----------
    calc_nfw(rbins, offsets=None, use_c=True, epsabs=0.1, epsrel=0.1)
        Generate Sigma and DeltaSigma NFW profiles for each cluster,
        optionally with miscentering offsets included.
    show(notebook=True)
        Display table of cluster information and mass-richness
        scaling relaton in use.
    massrich_parameters()
        Print a string showing the mass-richness scaling relation and
        current values of the normalization and slope.
    """
    def __init__(self, redshifts):
        if type(redshifts) != np.ndarray:
            redshifts = np.array(redshifts)
        if redshifts.ndim != 1:
            raise ValueError("Input redshift array must have 1 dimension.")
        if np.sum(redshifts < 0.) > 0:
            raise ValueError("Redshifts cannot be negative.")
        self.describe = "Ensemble of galaxy clusters and their properties."
        self.number = redshifts.shape[0]
        self._z = redshifts
        self._rho_crit = cosmo.critical_density(self._z)
        self._massrich_norm = 2.7*10**13 * units.Msun 
        self._massrich_slope = 1.4
        self._df = pd.DataFrame(self._z, columns=['z'])
        self._Dang_l = cosmo.angular_diameter_distance(self._z)
        self._m200 = None
        self._n200 = None
        self._r200 = None
        self._rs = None
        self._c200 = None
        self._deltac = None
        
    @property
    def n200(self):
        """Cluster richness values.

        If n200 is set directly, then mass m200 is calculated from n200
        using the mass-richness scaling relation specified by the
        parameters massrich_norm and massrich_slope. If m200 is set
        directly, then n200 is calculated from m200 using the same scaling
        relation. Changes to n200 will propagate to all mass-dependant
        variables.

        :property: Returns cluster richness values
        :property type: ndarray
        :setter: Sets cluster richness values
        :setter type: array_like
        """
        if self._n200 is None:
            raise AttributeError('n200 has not yet been initialized.')
        else:
            return self._n200

    @n200.setter
    def n200(self, richness):
        #Creates/updates values of cluster N200s & dependant variables.
        self._n200 = utils.check_units_and_type(richness, None,
                                                num = self.number)
        self._df['n200'] = pd.Series(self._n200, index = self._df.index)
        self._richness_to_mass()
        self._update_dependant_variables()

    @property
    def m200(self):
        """Cluster masses.

        Mass interior to a sphere of radius r200. If m200 is set directly,
        then richness n200 is calculated from m200 using the mass-richness
        scaling relation specified by the parameters massrich_norm and
        massrich_slope. If n200 is set directly, then m200 is calculated
        from n200 using the same scaling relation. Changes to m200 will
        propagate to all mass-dependant variables.

        :property: Returns cluster masses in Msun
        :type: Quantity (1D ndarray, with astropy.units of Msun)
        :setter: Sets cluster mass values in Msun
        :setter type: array_like        
        """
        if self._m200 is None:
            raise AttributeError('Attribute has not yet been initialized.')
        else:
            return self._m200

    @m200.setter
    def m200(self, mass):
        #Creates/updates values of cluster M200s & dependant variables.
        self._m200 = utils.check_units_and_type(mass, units.Msun,
                                                num = self.number)
        self._df['m200'] = pd.Series(self._m200, index = self._df.index)
        self._mass_to_richness()
        self._update_dependant_variables()
                
    def _richness_to_mass(self):
        #Calculates M_200 for simple power-law scaling relation
        #(with default parameters from arXiv:1409.3571)
        self._m200 = self._massrich_norm * ((self._n200/20.) **
                                            self._massrich_slope)
        self._df['m200'] = pd.Series(self._m200, index = self._df.index)
        
    def _mass_to_richness(self):
        #Calculates N_200 for simple power-law scaling relation.
        #Inverse of _richness_to_mass() function.
        n200 = 20. * (self._m200 /
                      self._massrich_norm)**(1. / self._massrich_slope)
        #note: units cancel but n200 is still a Quantity
        self._n200 = n200.value
        self._df['n200'] = pd.Series(self._n200, index = self._df.index)
        
    @property
    def z(self):
        """Cluster redshifts.

        :property: Returns cluster redshifts
        :property type: ndarray
        :setter: Sets cluster redshifts
        :setter type: array_like
        """
        return self._z

    @z.setter
    def z(self, redshifts):
        #Changes the values of the cluster z's and z-dependant variables.
        self._z = utils.check_units_and_type(redshifts, None,
                                             num = self.number)
        self._Dang_l = cosmo.angular_diameter_distance(self._z)
        self._df['z'] = pd.Series(self._z, index = self._df.index)
        self._rho_crit = cosmo.critical_density(self._z)
        if self._n200 is not None:
            self._update_dependant_variables()        

    def _update_dependant_variables(self):
        self._calculate_r200()
        self._calculate_concentrations()
        self._calculate_rs()
        #what else depends on z or m or?
                            
    @property
    def Dang_l(self):
        """Angular diameter distances to clusters.

        :property: Returns distances in Mpc
        :type: Quantity (1D ndarray, with astropy.units of Mpc)
        """
        return self._Dang_l        
        
    @property
    def dataframe(self):
        """Pandas DataFrame of cluster properties.

        :property: Returns DataFrame
        :type: pandas.core.frame.DataFrame
        """
        return self._df
       
    @property
    def massrich_norm(self):
        """Normalization of Mass-Richness relation:

        M200 = norm * (N200 / 20) ^ slope.

        Changes to massrich_norm will propagate to all mass-dependant
        variables.

        :property: Returns normalization in Msun
        :property type: Quantity (float, with astropy.units of Msun)
        :setter: Sets normalization in Msun
        :setter type: float (optionally in astropy.units of Msun)
        """
        return self._massrich_norm

    @massrich_norm.setter
    def massrich_norm(self, norm, rich_to_mass = True):
        self._massrich_norm = utils.check_units_and_type(norm, units.Msun,
                                                         is_scalar = True)
        #default behavior is to take current n200 -> m200
        # with new scaling relation
        if hasattr(self, 'n200'):
            if rich_to_mass == True:
                self._richness_to_mass()
            else:
                self._mass_to_richness()
        #TO DO: test above
        
    @property
    def massrich_slope(self):
        """Slope of Mass-Richness relation:

        M200 = norm * (N200 / 20) ^ slope.

        Changes to massrich_slope will propagate to all mass-dependant
        variables.

        :property: Returns slope
        :property type: float
        :setter: Sets slope
        :setter type: float
        """
        return self._massrich_slope

    @massrich_slope.setter
    def massrich_slope(self, slope, rich_to_mass = True):
        if type(slope) == float:
            self._massrich_slope = slope
        else:
            raise TypeError('Expecting input type as float')
        #default behavior is to take current n200 -> m200
        # with new scaling relation
        if hasattr(self, 'n200'):
            if rich_to_mass == True:
                self._richness_to_mass()
            else:
                self._mass_to_richness()
        #TO DO: test above

    def massrich_parameters(self):
        """Print values of M200-N200 scaling relation parameters."""
        print("\nMass-Richness Power Law: M200 = norm * (N200 / 20) ^ slope")
        print("   norm:", self._massrich_norm)
        print("   slope:", self._massrich_slope)
        
    def show(self, notebook = notebook_display):
        """Display cluster properties and scaling relation parameters."""
        print("\nCluster Ensemble:")
        if notebook == True:
            display(self._df)
        elif notebook == False:
            print(self._df)
        self.massrich_parameters()

    @property
    def r200(self):
        """Cluster Radii.

        r200 is the cluster radius within which the mean density is 200
        times the critical energy density of the universe at that z.

        :property: Returns r200 in Mpc
        :type: Quantity (1D ndarray, in astropy.units of Mpc)
        """
        if self._r200 is None:
            raise AttributeError('Attribute has not yet been initialized.')
        else:
            return self._r200
    
    @property
    def c200(self):
        """Cluster concentration parameters.

        c200 is calculated from m200 and z using the mass-concentration
        relation of Dutton & Maccio 2015. Note that c200 = r200/rs.

        :property: Returns c200
        :type: ndarray
        """
        if self._c200 is None:
            raise AttributeError('Attribute has not yet been initialized.')
        else:
            return self._c200
    
    @property
    def rs(self):
        """Cluster scale radii.

        :property: Returns scale radius in Mpc
        :type: Quantity (1D ndarray, in astropy.units of Mpc)
        """
        if self._rs is None:
            raise AttributeError('Attribute has not yet been initialized.')
        else:
            return self._rs

    @property
    def delta_c(self):
        """Characteristic overdensities of the cluster halos.

        :property: Returns characteristic overdensity
        :type: ndarray
        """        
        if self._deltac is None:
            raise AttributeError('Attribute has not yet been initialized.')
        else:
            return self._deltac
        
    def _calculate_r200(self):
        #calculate r200 from m200
        radius_200 = (3.*self._m200 / (800.*np.pi*self._rho_crit))**(1./3.)
        self._r200 = radius_200.to(units.Mpc)
        self._df['r200'] = pd.Series(self._r200, index = self._df.index)
        
    def _calculate_concentrations(self):
        #calculate c200 from m200 and z (using cofm.py)
        self._c200 = calc_c200(self._z,self._m200)
        self._df['c200'] = pd.Series(self._c200, index = self._df.index)
        self._calculate_deltac()
        
    def _calculate_rs(self):
        #cluster scale radius
        self._rs = self._r200 / self._c200
        self._df['rs'] = pd.Series(self._rs, index = self._df.index)

    def _calculate_deltac(self):
        #calculate concentration parameter from c200
        top = (200./3.)*self._c200**3.
        bottom = np.log(1.+self._c200)-(self._c200/(1.+self._c200))
        self._deltac = top/bottom
        self._df['delta_c'] = pd.Series(self._deltac, index = self._df.index)

    def calc_nfw(self, rbins, offsets = None, use_c = True,
                 epsabs=0.1, epsrel=0.1):
        """Calculates Sigma and DeltaSigma profiles.

        Generates the surface mass density (sigma_nfw attribute of parent
        object) and differential surface mass density (deltasigma_nfw
        attribute of parent object) profiles of each cluster, assuming a
        spherical NFW model. Optionally includes the effect of cluster
        miscentering offsets.
        
        Parameters
        ----------
        rbins : array_like
            Radial bins (in Mpc) for calculating cluster profiles. Should
            be 1D, optionally with astropy.units of Mpc.
        offsets : array_like, optional
            Parameter describing the width (in Mpc) of the Gaussian
            distribution of miscentering offsets. Should be 1D, optionally
            with astropy.units of Mpc.
        use_c : bool, optional    
            Sets whether to use the faster c implementation of calculation
            (use_c=True, default), or the Python version (use_c=False).
        epsabs, epsrel : float, optional
            Absolute and relative tolerances of the integration in the
            Python implementation of the miscentering calculations.
        """
        if offsets is None:
            self._sigoffset = np.zeros(self.number)*units.Mpc
        else:
            self._sigoffset = utils.check_units_and_type(offsets, units.Mpc,
                                                         num = self.number)

        self.rbins = utils.check_units_and_type(rbins, units.Mpc)


        if use_c:
            #--------
            #the old c way
            smdout = np.transpose(np.vstack(([self.rs],
                                            [self.delta_c],
                                            [self._rho_crit.to(units.Msun/
                                                            units.pc**3)],
                                            [self._sigoffset])))
            np.savetxt('smd_in1.dat', np.transpose(self.rbins), fmt='%15.8g')
            np.savetxt('smd_in2.dat', smdout, fmt='%15.8g')
            os.system('./smd_nfw')    #c program does the calculations
            sigma_nfw = np.loadtxt('sigma.dat') 
            deltasigma_nfw = np.loadtxt('deltasigma.dat')
            os.system('rm -f smd_in1.dat')
            os.system('rm -f smd_in2.dat')
            os.system('rm -f sigma.dat')
            os.system('rm -f deltasigma.dat')
            #--------

            #if offsets is None:
            self.sigma_nfw = sigma_nfw * units.Msun/(units.pc**2)
            self.deltasigma_nfw = deltasigma_nfw * units.Msun/(units.pc**2)
            
        else:
            #the python way
            smd = SurfaceMassDensity(self.rs,
                                     self.delta_c,
                                     self._rho_crit.to(units.Msun/units.pc**2/
                                                       units.Mpc),
                                     sig_offset = self._sigoffset,
                                     rbins = self.rbins)
            
            #optionally specify integration tolerances
            self.sigma_nfw = smd.sigma_nfw(epsabs=epsabs, epsrel=epsrel) 
            #self.sigma_nfw = smd.sigma_nfw()
            
            if offsets is None:           
                self.deltasigma_nfw = smd.deltasigma_nfw() #not yet implemented for offsets
            #else:
                #raise ValueError("Python does not yet calculate offset profiles. Use the c option.\n")



                

