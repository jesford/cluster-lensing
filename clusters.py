from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from astropy.cosmology import Planck13 as cosmo
from astropy import units

import sys
sys.path.insert(1,'/Users/jesford/astrophysics/cofm') #temporary path adjust
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

class ClusterEnsemble():
    """Ensemble of galaxy clusters and their properties."""
    def __init__(self, redshifts):
        if type(redshifts) != np.ndarray:
            redshifts = np.array(redshifts)
        if redshifts.ndim != 1:
            raise ValueError("Input redshift array must have 1 dimension.")
        self.describe = "Ensemble of galaxy clusters and their properties."
        self.number = redshifts.shape[0]
        self.z = redshifts
        self._rho_crit = cosmo.critical_density(self.z)
        self._massrich_norm = 2.7*10**13 * units.Msun 
        self._massrich_slope = 1.4
        self._df = pd.DataFrame(self.z, columns=['z'])
              
    def update_richness(self, richness):
        if type(richness) != np.ndarray:
            richness = np.array(richness)
        if richness.ndim != 1:
            raise ValueError("Input richness array must have 1 dimension.")
        if richness.shape[0] == self.number:
            self.n200 = richness
            self._df['n200'] = pd.Series(self.n200, index = self._df.index)
            self._richness_to_mass()
        else:
            raise ValueError("Input richness array must be same \
            length as current cluster ensemble.")

    def _richness_to_mass(self):
        """Calculate M_200 for simple power-law scaling relation
        (with default parameters from arXiv:1409.3571)."""
        self.m200 = self._massrich_norm * ((self.n200/20.) ** self._massrich_slope)
        self._df['m200'] = pd.Series(self.m200, index = self._df.index)
        self._update_dependant_variables()

    def update_z(self, redshifts):
        if type(redshifts) != np.ndarray:
            redshifts = np.array(redshifts)
        if redshifts.ndim != 1:
            raise ValueError("Input richness array must have 1 dimension.")
        if redshifts.shape[0] == self.number:
            self.z = redshifts
            self._df['z'] = pd.Series(self.z, index = self._df.index)
            self._rho_crit = cosmo.critical_density(self.z)
            self._update_dependant_variables()
        else:
            raise ValueError("Input redshifts array must be same \
            length as current cluster ensemble.")
        
    def _update_dependant_variables(self):
        self._r200()
        self._c200()
        self._rs()
        #what else depends on z or m or?
    
    def massrich_parameters(self):
        print("\nMass-Richness Power Law: M200 = norm * (N200 / 20) ^ slope")
        print("   norm:", self._massrich_norm)
        print("   slope:", self._massrich_slope)
        
    def update_massrichrelation(self, norm = None, slope = None):
        if norm != None:
            if type(norm) == float:
                self._massrich_norm = norm * units.Msun
            else:
                raise TypeError("Input norm must be of type float, in units \
                of solar mass.")
        if slope != None:
            self._massrich_slope = slope
        self._richness_to_mass()
    
    def show(self, notebook = notebook_display):
        print("\nCluster Ensemble:")
        if notebook == True:
            display(self._df)
        elif notebook == False:
            print(self._df)
        self.massrich_parameters()

    def _r200(self):
        radius_200 = (3.*self.m200 / (800.*np.pi*self._rho_crit))**(1./3.)
        self.r200 = radius_200.to(units.Mpc)
        self._df['r200'] = pd.Series(self.r200, index = self._df.index)
        
    def _c200(self):
        """Use c(M) from Dutton & Maccio 2014."""
        self.c200 = calc_c200(self.z,self.m200)
        self._df['c200'] = pd.Series(self.c200, index = self._df.index)
        self._delta_c()
        
    def _rs(self):
        """Cluster scale radius."""
        self.rs = self.r200 / self.c200
        self._df['rs'] = pd.Series(self.rs, index = self._df.index)

    def _delta_c(self):
        top = (200./3.)*self.c200**3.
        bottom = np.log(1.+self.c200)-(self.c200/(1.+self.c200))
        self.delta_c = top/bottom
        self._df['delta_c'] = pd.Series(self.delta_c, index = self._df.index)
