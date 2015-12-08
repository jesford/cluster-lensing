from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from astropy.cosmology import Planck13 as cosmo
from astropy import units
import os

from cofm import c_DuttonMaccio as calc_c200

try:
    from IPython.display import display
    notebook_display = True
except:
    notebook_display = False

#default parameters
h = cosmo.H0.value
Om_M = cosmo.Om0
Om_L = 1. - Om_M

class ClusterEnsemble(object):
    """Ensemble of galaxy clusters and their properties."""
    def __init__(self, redshifts):
        if type(redshifts) != np.ndarray:
            redshifts = np.array(redshifts)
        if redshifts.ndim != 1:
            raise ValueError("Input redshift array must have 1 dimension.")
        if np.sum(redshifts < 0.) > 0:
            raise ValueError("Redshifts cannot be negative.")
        self.describe = "Ensemble of galaxy clusters and their properties."
        self.number = redshifts.shape[0]
        self.z = redshifts
        self._rho_crit = cosmo.critical_density(self.z)
        self._massrich_norm = 2.7*10**13 * units.Msun 
        self._massrich_slope = 1.4
        self._df = pd.DataFrame(self.z, columns=['z'])
        self.Dang_l = cosmo.angular_diameter_distance(self.z) 
              
    def update_richness(self, richness):
        """Creates/updates values of cluster N200s & dependant variables."""
        self.n200 = self._check_input_array(richness)
        self._df['n200'] = pd.Series(self.n200, index = self._df.index)
        self._richness_to_mass()

    def _richness_to_mass(self):
        #Calculates M_200 for simple power-law scaling relation
        #(with default parameters from arXiv:1409.3571)
        self.m200 = self._massrich_norm * ((self.n200/20.) ** self._massrich_slope)
        self._df['m200'] = pd.Series(self.m200, index = self._df.index)
        self._update_dependant_variables()

    def update_z(self, redshifts):
        """Changes the values of the cluster z's and z-dependant variables."""
        self.z = self._check_input_array(redshifts)
        self.Dang_l = cosmo.angular_diameter_distance(self.z)
        self._df['z'] = pd.Series(self.z, index = self._df.index)
        self._rho_crit = cosmo.critical_density(self.z)
        self._update_dependant_variables()

    def _check_input_array(self, arr):
        #confirm input array matches size/type of clusters
        if type(arr) != np.ndarray:
            arr = np.array(arr)
        if arr.ndim != 1:
            raise ValueError("Input array must have 1 dimension.")
        elif arr.shape[0] != self.number:
            raise ValueError("Input array must be same length as current \
                              cluster ensemble.")
        if np.sum(arr < 0.) > 0:
            raise ValueError("Input array values cannot be negative.")
        else:
            return arr

    def _update_dependant_variables(self):
        self._r200()
        self._c200()
        self._rs()
        #what else depends on z or m or?
    
    def massrich_parameters(self):
        """Print values of M200-N200 scaling relation parameters."""
        print("\nMass-Richness Power Law: M200 = norm * (N200 / 20) ^ slope")
        print("   norm:", self._massrich_norm)
        print("   slope:", self._massrich_slope)
        
    def update_massrichrelation(self, norm = None, slope = None):
        """Updates scaling relation parameters & dependant variables."""
        if norm is not None:
            if type(norm) == float:
                self._massrich_norm = norm * units.Msun
            else:
                raise TypeError("Input norm must be of type float, in units \
                of solar mass.")
        if slope is not None:
            if type(slope) == float:
                self._massrich_slope = slope
            else:
                raise TypeError("Input slope must be of type float.")
        if hasattr(self, 'n200'):
            self._richness_to_mass()
    
    def show(self, notebook = notebook_display):
        """Display table of cluster properties."""
        print("\nCluster Ensemble:")
        if notebook == True:
            display(self._df)
        elif notebook == False:
            print(self._df)
        self.massrich_parameters()

    def _r200(self):
        #calculate r200 from m200
        radius_200 = (3.*self.m200 / (800.*np.pi*self._rho_crit))**(1./3.)
        self.r200 = radius_200.to(units.Mpc)
        self._df['r200'] = pd.Series(self.r200, index = self._df.index)
        
    def _c200(self):
        #calculate c200 from m200 and z (using cofm.py)
        self.c200 = calc_c200(self.z,self.m200)
        self._df['c200'] = pd.Series(self.c200, index = self._df.index)
        self._delta_c()
        
    def _rs(self):
        #cluster scale radius
        self.rs = self.r200 / self.c200
        self._df['rs'] = pd.Series(self.rs, index = self._df.index)

    def _delta_c(self):
        #calculate concentration parameter from c200
        top = (200./3.)*self.c200**3.
        bottom = np.log(1.+self.c200)-(self.c200/(1.+self.c200))
        self.delta_c = top/bottom
        self._df['delta_c'] = pd.Series(self.delta_c, index = self._df.index)

    def calc_nfw(self, rbins, offsets = None):
        """Calculates Sigma and DeltaSigma NFW profiles of each cluster."""
        if offsets is None:
            self._sigoffset = np.zeros(self.number)*units.Mpc
        else:
            self._sigoffset = self._check_input_array(offsets)*units.Mpc

        self.rbins = rbins * units.Mpc

        #--------
        #the old way
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

        if offsets is None:
            self.sigma_nfw = sigma_nfw * units.Msun/(units.pc**2)
            self.deltasigma_nfw = deltasigma_nfw * units.Msun/(units.pc**2)
        else:
            self.sigma_offset = sigma_nfw * units.Msun/(units.pc**2)
            self.deltasigma_offset = deltasigma_nfw * units.Msun/(units.pc**2)


