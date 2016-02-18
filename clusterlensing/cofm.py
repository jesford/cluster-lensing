"""Calculate halo concentration from mass and redshift.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy import integrate
from astropy.cosmology import Planck13 as cosmo

# default cosmological parameters
h = cosmo.h
Om_M = cosmo.Om0
Om_L = 1. - Om_M


def _check_inputs(z, m):
    """Check inputs are arrays of same length or array and a scalar."""
    try:
        nz = len(z)
        z = np.array(z)
    except TypeError:
        z = np.array([z])
        nz = len(z)
    try:
        nm = len(m)
        m = np.array(m)
    except TypeError:
        m = np.array([m])
        nm = len(m)

    if (z < 0).any() or (m < 0).any():
        raise ValueError('z and m must be positive')

    if nz != nm and nz > 1 and nm > 1:
        raise ValueError('z and m arrays must be either equal in length, \
                          OR of different length with one of length 1.')

    else:
        if type(z) != np.ndarray:
            z = np.array(z)
        if type(m) != np.ndarray:
            m = np.array(m)

        return z, m


def c_Prada(z, m, h=h, Om_M=Om_M, Om_L=Om_L):
    """Concentration from c(M) relation published in Prada et al. (2012).

    Parameters
    ----------
    z : float or array_like
        Redshift(s) of halos.
    m : float or array_like
        Mass(es) of halos (m200 definition), in units of solar masses.
    h : float, optional
        Hubble parameter. Default is from Planck13.
    Om_M : float, optional
        Matter density parameter. Default is from Planck13.
    Om_L : float, optional
        Cosmological constant density parameter. Default is from Planck13.

    Returns
    ----------
    ndarray
        Concentration values (c200) for halos.

    Notes
    ----------
    This c(M) relation is somewhat controversial, due to its upturn in
    concentration for high masses (normally we expect concentration to
    decrease with increasing mass). See the reference below for discussion.

    References
    ----------
    Calculation based on results of N-body simulations presented in:

    F. Prada, A.A. Klypin, A.J. Cuesta, J.E. Betancort-Rijo, and J.
    Primack, "Halo concentrations in the standard Lambda cold dark matter
    cosmology," Monthly Notices of the Royal Astronomical Society, Volume
    423, Issue 4, pp. 3018-3030, 2012.
    """

    z, m = _check_inputs(z, m)

    # EQ 13
    x = (1. / (1. + z)) * (Om_L / Om_M)**(1. / 3.)

    # EQ 12
    intEQ12 = np.zeros(len(x))  # integral
    for i in range(len(x)):
        # v is integration variable
        temp = integrate.quad(lambda v: (v / (1 + v**3.))**(1.5), 0, x[i])
        intEQ12[i] = temp[0]

    Da = 2.5 * ((Om_M / Om_L)**(1. / 3.)) * (np.sqrt(1. + x**3.) /
                                             (x**(1.5))) * intEQ12

    # EQ 23
    y = (1.e+12) / (h * m)
    sigma = Da * (16.9 * y**0.41) / (1. + (1.102 * y**0.2) + (6.22 * y**0.333))

    # EQ 21 & 22 (constants)
    c0 = 3.681
    c1 = 5.033
    alpha = 6.948
    x0 = 0.424
    s0 = 1.047  # sigma_0^-1
    s1 = 1.646  # sigma_1^-1
    beta = 7.386
    x1 = 0.526

    # EQ 19 & 20
    cmin = c0 + (c1 - c0) * ((1. / np.pi) * np.arctan(alpha * (x - x0)) + 0.5)
    smin = s0 + (s1 - s0) * ((1. / np.pi) * np.arctan(beta * (x - x1)) + 0.5)

    # EQ 18
    cmin1393 = c0 + (c1 - c0) * ((1. / np.pi) * np.arctan(alpha *
                                                          (1.393 - x0)) + 0.5)
    smin1393 = s0 + (s1 - s0) * ((1. / np.pi) * np.arctan(beta *
                                                          (1.393 - x1)) + 0.5)
    B0 = cmin / cmin1393
    B1 = smin / smin1393

    # EQ 15
    sigma_prime = B1 * sigma

    # EQ 17
    A = 2.881
    b = 1.257
    c = 1.022
    d = 0.06

    # EQ 16
    Cs = A * ((sigma_prime / b)**c + 1.) * np.exp(d / (sigma_prime**2.))

    # EQ 14
    concentration = B0 * Cs

    return concentration


def c_DuttonMaccio(z, m, h=h):
    """Concentration from c(M) relation in Dutton & Maccio (2014).

    Parameters
    ----------
    z : float or array_like
        Redshift(s) of halos.
    m : float or array_like
        Mass(es) of halos (m200 definition), in units of solar masses.
    h : float, optional
        Hubble parameter. Default is from Planck13.

    Returns
    ----------
    ndarray
        Concentration values (c200) for halos.

    References
    ----------
    Calculation from Planck-based results of simulations presented in:

    A.A. Dutton & A.V. Maccio, "Cold dark matter haloes in the Planck era:
    evolution of structural parameters for Einasto and NFW profiles,"
    Monthly Notices of the Royal Astronomical Society, Volume 441, Issue 4,
    p.3359-3374, 2014.
    """

    z, m = _check_inputs(z, m)

    a = 0.52 + 0.385 * np.exp(-0.617 * (z**1.21))  # EQ 10
    b = -0.101 + 0.026 * z                         # EQ 11

    logc200 = a + b * np.log10(m * h / (10.**12))  # EQ 7

    concentration = 10.**logc200

    return concentration


def c_Duffy(z, m, h=h):
    """Concentration from c(M) relation published in Duffy et al. (2008).

    Parameters
    ----------
    z : float or array_like
        Redshift(s) of halos.
    m : float or array_like
        Mass(es) of halos (m200 definition), in units of solar masses.
    h : float, optional
        Hubble parameter. Default is from Planck13.

    Returns
    ----------
    ndarray
        Concentration values (c200) for halos.

    References
    ----------
    Results from N-body simulations using WMAP5 cosmology, presented in:

    A.R. Duffy, J. Schaye, S.T. Kay, and C. Dalla Vecchia, "Dark matter
    halo concentrations in the Wilkinson Microwave Anisotropy Probe year 5
    cosmology," Monthly Notices of the Royal Astronomical Society, Volume
    390, Issue 1, pp. L64-L68, 2008.

    This calculation uses the parameters corresponding to the NFW model,
    the '200' halo definition, and the 'full' sample of halos spanning
    z = 0-2. This means the values of fitted parameters (A,B,C) = (5.71,
    -0.084,-0.47) in Table 1 of Duffy et al. (2008).
    """

    z, m = _check_inputs(z, m)

    M_pivot = 2.e12 / h  # [M_solar]

    A = 5.71
    B = -0.084
    C = -0.47

    concentration = A * ((m / M_pivot)**B) * (1 + z)**C

    return concentration
