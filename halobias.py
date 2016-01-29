import numpy as np
from astropy.cosmology import Planck13 as cosmo

#default parameters
h = cosmo.h
Om_M = cosmo.Om0
Om_L = 1. - Om_M

def bias(mass_halo, z_halo, h = h, Om_M = Om_M, Om_L = Om_L):
    """Calculate halo bias, from Seljak & Warren 2004.

    Parameters
    ----------
    mass_halo : ndarray or float
        Halo mass to calculate bias for.
    z_halo : ndarray or float
        Halo z, same type and size as mass_halo.
    h : float, optional
        Hubble parameter, defaults to astropy.cosmology.Planck13.h
    Om_M : float, optional
        Fractional matter density, defaults to 
        astropy.cosmology.Planck13.Om0
    Om_L : float, optional
        Fractional dark energy density, defaults to 1-Om_M.

    Returns
    ----------
    ndarray or float
        Returns the halo bias, of same type and size as input mass_halo and
        z_halo. Calculated according to Seljak & Warren 2004.
    """
    M_nl_0 = (8.73/h) * 10.**12.         #nonlinear mass today [M_sun]
    M_nl = M_nl_0 * (Om_M + Om_L/((1.+z_halo)**3.))  #scaled to z_lens
    x = mass_halo/M_nl
    b = 0.53 + 0.39*(x**0.45) + 0.13/(40.*x + 1.) + (5.e-4)*(x**1.5)
    return b
