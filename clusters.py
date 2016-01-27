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
    """Ensemble of galaxy clusters and their properties."""
    def __init__(self, redshifts):
        """ Initialize a ClusterEnsemble object
        
        Parameters
        ----------
        z : Numpy 1D array or list
            Redshifts for each cluster in the sample.
        

        Attributes
        ----------
        z : Numpy 1D array or numpy.ndarray?
            Cluster redshifts.
        n200 : Numpy 1D array
            Cluster richness.
        m200 : Numpy 1D array, with astropy.units of Msun...
               or astropy.units.quantity.Quantity?
            Cluster masses in units of solar masses.
        c200 : Numpy 1D array
            Cluster concentration parameters.
        delta_c : Numpy 1D array
            Characteristic overdensities of the cluster halos.
        r200 : Numpy 1D array, with astropy.units of Mpc
            Cluster radii in units of Mpc.
        rs : Numpy 1D array, with astropy.units of Mpc
            Cluster scale radii in units of Mpc.            
        Dang_l : Numpy 1D array, with astropy.units of Mpc
            Angular diameter distances from z=0 to z, in units of Mpc.
        dataframe : pandas.core.frame.DataFrame
            Cluster information in tabular form.
        describe : str
            Short description of the ClusterEnsemble object.
        number : int
            Number of clusters in the sample.
        
        
        Methods
        ----------
        show(notebook=True)
            Display table of cluster information and mass-richness
            scaling relaton in use.
        calc_nfw(rbins, offsets=None, use_c=True, epsabs=0.1,
                 epsrel=0.1)
            Generate Sigma and DeltaSigma NFW profiles for each cluster,
            optionally with miscentering offsets included.
        """
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
        """Cluster richness values."""
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

        
    def _richness_to_mass(self):
        #Calculates M_200 for simple power-law scaling relation
        #(with default parameters from arXiv:1409.3571)
        self._m200 = self._massrich_norm * ((self._n200/20.) **
                                            self._massrich_slope)
        self._df['m200'] = pd.Series(self._m200, index = self._df.index)
        self._update_dependant_variables()


    @property
    def z(self):
        """Cluster redshifts."""
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
        
    #note: user can access, but not modify, functions that are ONLY decorated
    # as properties; attempting to modify them will raise an AttributeError
        
    @property
    def Dang_l(self):
        return self._Dang_l

    @property
    def m200(self):
        if self._m200 is None:
            raise AttributeError('Attribute has not yet been initialized.')
        else:
            return self._m200

    @property
    def dataframe(self):
        return self._df


    def _update_dependant_variables(self):
        self._calculate_r200()
        self._calculate_c200()
        self._calculate_rs()
        #what else depends on z or m or?

        
    @property
    def massrich_norm(self):
        """Normalization of Mass-Richness relation:
        M200 = norm * (N200 / 20) ^ slope."""
        return self._massrich_norm

    @massrich_norm.setter
    def massrich_norm(self, norm):
        self._massrich_norm = utils.check_units_and_type(norm, units.Msun,
                                                         is_scalar = True)
        if hasattr(self, 'n200'):
            self._richness_to_mass()
        
    @property
    def massrich_slope(self):
        """Slope of Mass-Richness relation:
        M200 = norm * (N200 / 20) ^ slope."""
        return self._massrich_slope

    @massrich_slope.setter
    def massrich_slope(self, slope):
        if type(slope) == float:
            self._massrich_slope = slope
        else:
            raise TypeError('Expecting input type as float')
        if hasattr(self, 'n200'):
            self._richness_to_mass()

    #TO DO: should this be a property or not??
    def massrich_parameters(self):
        """Print values of M200-N200 scaling relation parameters."""
        print("\nMass-Richness Power Law: M200 = norm * (N200 / 20) ^ slope")
        print("   norm:", self._massrich_norm)
        print("   slope:", self._massrich_slope)
        
    #TO DO: should this be a property or not??
    def show(self, notebook = notebook_display):
        """Display table of cluster properties."""
        print("\nCluster Ensemble:")
        if notebook == True:
            display(self._df)
        elif notebook == False:
            print(self._df)
        self.massrich_parameters()

    @property
    def r200(self):
        if self._r200 is None:
            raise AttributeError('Attribute has not yet been initialized.')
        else:
            return self._r200
    
    @property
    def c200(self):
        if self._c200 is None:
            raise AttributeError('Attribute has not yet been initialized.')
        else:
            return self._c200
    
    @property
    def rs(self):
        if self._rs is None:
            raise AttributeError('Attribute has not yet been initialized.')
        else:
            return self._rs

    @property
    def delta_c(self):
        if self._deltac is None:
            raise AttributeError('Attribute has not yet been initialized.')
        else:
            return self._deltac
        
    def _calculate_r200(self):
        #calculate r200 from m200
        radius_200 = (3.*self._m200 / (800.*np.pi*self._rho_crit))**(1./3.)
        self._r200 = radius_200.to(units.Mpc)
        self._df['r200'] = pd.Series(self._r200, index = self._df.index)
        
    def _calculate_c200(self):
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
        """Calculates Sigma and DeltaSigma NFW profiles of each cluster."""
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



                

