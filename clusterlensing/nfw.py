"""NFW profiles for shear and magnification.

Surface mass density and differential surface mass density calculations
for NFW dark matter halos, with and without the effects of miscentering
offsets.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from astropy import units
from scipy.integrate import simps

from clusterlensing import utils


def _set_dimensionless_radius(self, radii=None, integration=False):
    if radii is None:
        radii = self._rbins  # default radii

    # calculate x = radii / rs
    if integration is True:
        # radii is a 3D matrix (r_eq13 <-> numTh,numRoff,numRbins)
        d0, d1, d2 = radii.shape[0], radii.shape[1], radii.shape[2]

        # with each cluster's rs, we now have a 4D matrix:
        radii_4D = radii.reshape(d0, d1, d2, 1)
        rs_4D = self._rs.reshape(1, 1, 1, self._nlens)
        x = radii_4D / rs_4D

    else:
        # 1D array of radii and clusters, reshape & broadcast
        radii_repeated = radii.reshape(1, self._nbins)
        rs_repeated = self._rs.reshape(self._nlens, 1)
        x = radii_repeated / rs_repeated

    x = x.value
    if 0. in x:
        x[np.where(x == 0.)] = 1.e-10  # hack to avoid infs in sigma

    # dimensionless radius
    self._x = x

    # set the 3 cases of dimensionless radius x
    self._x_small = np.where(self._x < 1. - 1.e-6)
    self._x_big = np.where(self._x > 1. + 1.e-6)
    self._x_one = np.where(np.abs(self._x - 1) <= 1.e-6)


class SurfaceMassDensity(object):
    """Calculate NFW profiles for Sigma and Delta-Sigma.

    Parameters
    ----------
    rs : array_like
        Scale radii (in Mpc) for every halo. Should be 1D, optionally with
        astropy.units of Mpc.
    delta_c : array_like
        Characteristic overdensities for every halo. Should be 1D and have
        the same length as rs.
    rho_crit : array_like
        Critical energy density of the universe (in Msun/Mpc/pc^2) at every
        halo z. Should be 1D, optionally with astropy.units of
        Msun/Mpc/(pc**2), and have the same length as rs.
    offsets : array_like, optional
        Width of the Gaussian distribution of miscentering offsets (in
        Mpc), for every cluster halo. Should be 1D, optionally with
        astropy.units of Mpc, and have the same length as rs. (Note: it is
        common to use the same value for every halo, implying that they are
        drawn from the same offset distribution).
    rbins : array_like, optional
        Radial bins (in Mpc) at which profiles will be calculated. Should
        be 1D, optionally with astropy.units of Mpc.

    Other Parameters
    -------------------
    numTh : int, optional
        Number of bins to use for integration over theta, for calculating
        offset profiles (no effect for offsets=None). Default 200.
    numRoff : int, optional
        Number of bins to use for integration over R_off, for calculating
        offset profiles (no effect for offsets=None). Default 200.
    numRinner : int, optional
        Number of bins at r < min(rbins) to use for integration over
        Sigma(<r), for calculating DeltaSigma (no effect for Sigma ever,
        and no effect for DeltaSigma if offsets=None). Default 20.
    factorRouter : int, optional
        Factor increase over number of rbins, at min(r) < r < max(r), of
        bins that will be used at for integration over Sigma(<r), for
        calculating DeltaSigma (no effect for Sigma, and no effect for
        DeltaSigma if offsets=None). Default 3.

    Methods
    ----------
    sigma_nfw()
        Calculate surface mass density Sigma.
    deltasigma_nfw()
        Calculate differential surface mass density DeltaSigma.

    See Also
    ----------
    ClusterEnsemble : Parameters and profiles for a sample of clusters.
        This class provides an interface to SurfaceMassDensity, and tracks
        a DataFrame of parameters as well as nfw profiles for many clusters
        at once, only requiring the user to specify cluster z and richness,
        at a minimum.

    References
    ----------
    Sigma and DeltaSigma are calculated using the formulas given in:

    C.O. Wright and T.G. Brainerd, "Gravitational Lensing by NFW Halos,"
    The Astrophysical Journal, Volume 534, Issue 1, pp. 34-40 (2000).

    The offset profiles are calculated using formulas given, e.g., in
    Equations 11-15 of:

    J. Ford, L. Van Waerbeke, M. Milkeraitis, et al., "CFHTLenS: a weak
    lensing shear analysis of the 3D-Matched-Filter galaxy clusters,"
    Monthly Notices of the Royal Astronomical Society, Volume 447, Issue 2,
    p.1304-1318 (2015).
    """
    def __init__(self, rs, delta_c, rho_crit, offsets=None, rbins=None,
                 numTh=200, numRoff=200, numRinner=20, factorRouter=3):
        if rbins is None:
            rmin, rmax = 0.1, 5.
            rbins = np.logspace(np.log10(rmin), np.log10(rmax), num=50)
            self._rbins = rbins * units.Mpc
        else:
            # check rbins input units & type
            self._rbins = utils.check_units_and_type(rbins, units.Mpc)

        # check input units & types
        self._rs = utils.check_units_and_type(rs, units.Mpc)
        self._delta_c = utils.check_units_and_type(delta_c, None)
        self._rho_crit = utils.check_units_and_type(rho_crit, units.Msun /
                                                    units.Mpc / (units.pc**2))

        self._numRoff = utils.check_units_and_type(numRoff, None,
                                                   is_scalar=True)
        self._numTh = utils.check_units_and_type(numTh, None, is_scalar=True)
        self._numRinner = utils.check_units_and_type(numRinner, None,
                                                     is_scalar=True)
        self._factorRouter = utils.check_units_and_type(factorRouter, None,
                                                        is_scalar=True)
        # check numbers of bins are all positive
        if (numRoff <= 0) or (numTh <= 0) or (numRinner <= 0) or \
           (factorRouter <= 0):
                raise ValueError('Require numbers of bins > 0')

        self._nbins = self._rbins.shape[0]
        self._nlens = self._rs.shape[0]

        if offsets is not None:
            self._sigmaoffset = utils.check_units_and_type(offsets, units.Mpc)
            utils.check_input_size(self._sigmaoffset, self._nlens)
        else:
            self._sigmaoffset = offsets

        # check array sizes are compatible
        utils.check_input_size(self._rs, self._nlens)
        utils.check_input_size(self._delta_c, self._nlens)
        utils.check_input_size(self._rho_crit, self._nlens)
        utils.check_input_size(self._rbins, self._nbins)

        rs_dc_rcrit = self._rs * self._delta_c * self._rho_crit
        self._rs_dc_rcrit = rs_dc_rcrit.reshape(self._nlens,
                                                1).repeat(self._nbins, 1)

        # set self._x, self._x_big, self._x_small, self._x_one
        _set_dimensionless_radius(self)

    def sigma_nfw(self):
        """Calculate NFW surface mass density profile.

        Generate the surface mass density profiles of each cluster halo,
        assuming a spherical NFW model. Optionally includes the effect of
        cluster miscentering offsets, if the parent object was initialized
        with offsets.

        Returns
        ----------
        Quantity
            Surface mass density profiles (ndarray, in astropy.units of
            Msun/pc/pc). Each row corresponds to a single cluster halo.
        """
        def _centered_sigma(self):
            # perfectly centered cluster case

            # calculate f
            bigF = np.zeros_like(self._x)
            f = np.zeros_like(self._x)

            numerator_arg = ((1. / self._x[self._x_small]) +
                             np.sqrt((1. / (self._x[self._x_small]**2)) - 1.))
            denominator = np.sqrt(1. - (self._x[self._x_small]**2))
            bigF[self._x_small] = np.log(numerator_arg) / denominator

            bigF[self._x_big] = (np.arccos(1. / self._x[self._x_big]) /
                                 np.sqrt(self._x[self._x_big]**2 - 1.))

            f = (1. - bigF) / (self._x**2 - 1.)
            f[self._x_one] = 1. / 3.
            if np.isnan(np.sum(f)) or np.isinf(np.sum(f)):
                print('\nERROR: f is not all real\n')

            # calculate & return centered profiles
            if f.ndim == 2:
                sigma = 2. * self._rs_dc_rcrit * f
            else:
                rs_dc_rcrit_4D = self._rs_dc_rcrit.T.reshape(1, 1,
                                                             f.shape[2],
                                                             f.shape[3])
                sigma = 2. * rs_dc_rcrit_4D * f

            return sigma

        def _offset_sigma(self):

            # size of "x" arrays to integrate over
            numRoff = self._numRoff
            numTh = self._numTh

            numRbins = self._nbins
            maxsig = self._sigmaoffset.value.max()

            # inner/outer bin edges
            roff_1D = np.linspace(0., 4. * maxsig, numRoff)
            theta_1D = np.linspace(0., 2. * np.pi, numTh)
            rMpc_1D = self._rbins.value

            # reshape for broadcasting: (numTh,numRoff,numRbins)
            theta = theta_1D.reshape(numTh, 1, 1)
            roff = roff_1D.reshape(1, numRoff, 1)
            rMpc = rMpc_1D.reshape(1, 1, numRbins)

            r_eq13 = np.sqrt(rMpc ** 2 + roff ** 2 -
                             2. * rMpc * roff * np.cos(theta))

            # 3D array r_eq13 -> 4D dimensionless radius (nlens)
            _set_dimensionless_radius(self, radii=r_eq13, integration=True)

            sigma = _centered_sigma(self)
            inner_integrand = sigma.value / (2. * np.pi)

            # INTEGRATE OVER theta
            sigma_of_RgivenRoff = simps(inner_integrand, x=theta_1D, axis=0,
                                        even='first')

            # theta is gone, now dimensions are: (numRoff,numRbins,nlens)
            sig_off_3D = self._sigmaoffset.value.reshape(1, 1, self._nlens)
            roff_v2 = roff_1D.reshape(numRoff, 1, 1)
            PofRoff = (roff_v2 / (sig_off_3D**2) *
                       np.exp(-0.5 * (roff_v2 / sig_off_3D)**2))

            dbl_integrand = sigma_of_RgivenRoff * PofRoff

            # INTEGRATE OVER Roff
            # (integration axis=0 after theta is gone).
            sigma_smoothed = simps(dbl_integrand, x=roff_1D, axis=0,
                                   even='first')

            # reset _x to correspond to input rbins (default)
            _set_dimensionless_radius(self)

            sigma_sm = np.array(sigma_smoothed.T) * units.solMass / units.pc**2

            return sigma_sm

        if self._sigmaoffset is None:
            finalsigma = _centered_sigma(self)
        elif np.abs(self._sigmaoffset).sum() == 0:
            finalsigma = _centered_sigma(self)
        else:
            finalsigma = _offset_sigma(self)
            self._sigma_sm = finalsigma

        return finalsigma

    def deltasigma_nfw(self):
        """Calculate NFW differential surface mass density profile.

        Generate the surface mass density profiles of each cluster halo,
        assuming a spherical NFW model. Currently calculates centered
        profiles ONLY; DOES NOT have the miscentering implemented.

        Returns
        ----------
        Quantity
            Differential surface mass density profiles (ndarray, in
            astropy.units of Msun/pc/pc). Each row corresponds to a single
            cluster halo.
        """
        def _centered_dsigma(self):
            # calculate g

            firstpart = np.zeros_like(self._x)
            secondpart = np.zeros_like(self._x)
            g = np.zeros_like(self._x)

            small_1a = 4. / self._x[self._x_small]**2
            small_1b = 2. / (self._x[self._x_small]**2 - 1.)
            small_1c = np.sqrt(1. - self._x[self._x_small]**2)
            firstpart[self._x_small] = (small_1a + small_1b) / small_1c

            big_1a = 8. / (self._x[self._x_big]**2 *
                           np.sqrt(self._x[self._x_big]**2 - 1.))
            big_1b = 4. / ((self._x[self._x_big]**2 - 1.)**1.5)
            firstpart[self._x_big] = big_1a + big_1b

            small_2a = np.sqrt((1. - self._x[self._x_small]) /
                               (1. + self._x[self._x_small]))
            secondpart[self._x_small] = np.log((1. + small_2a) /
                                               (1. - small_2a))

            big_2a = self._x[self._x_big] - 1.
            big_2b = 1. + self._x[self._x_big]
            secondpart[self._x_big] = np.arctan(np.sqrt(big_2a / big_2b))

            both_3a = (4. / (self._x**2)) * np.log(self._x / 2.)
            both_3b = 2. / (self._x**2 - 1.)
            g = firstpart * secondpart + both_3a - both_3b

            g[self._x_one] = (10. / 3.) + 4. * np.log(0.5)

            if np.isnan(np.sum(g)) or np.isinf(np.sum(g)):
                print('\nERROR: g is not all real\n', g)

            # calculate & return centered profile
            deltasigma = self._rs_dc_rcrit * g

            return deltasigma

        def _offset_dsigma(self):
            original_rbins = self._rbins.value

            # if offset sigma was already calculated, use it!
            try:
                sigma_sm_rbins = self._sigma_sm
            except AttributeError:
                sigma_sm_rbins = self.sigma_nfw()

            innermost_sampling = 1.e-10  # stable for anything below 1e-5
            inner_prec = self._numRinner
            r_inner = np.linspace(innermost_sampling,
                                  original_rbins.min(),
                                  endpoint=False, num=inner_prec)
            outer_prec = self._factorRouter * self._nbins
            r_outer = np.linspace(original_rbins.min(),
                                  original_rbins.max(),
                                  endpoint=False, num=outer_prec + 1)[1:]
            r_ext_unordered = np.hstack([r_inner, r_outer, original_rbins])
            r_extended = np.sort(r_ext_unordered)

            # set temporary extended rbins, nbins, x, rs_dc_rcrit array
            self._rbins = r_extended * units.Mpc
            self._nbins = self._rbins.shape[0]
            _set_dimensionless_radius(self)  # uses _rbins, _nlens
            rs_dc_rcrit = self._rs * self._delta_c * self._rho_crit
            self._rs_dc_rcrit = rs_dc_rcrit.reshape(self._nlens,
                                                    1).repeat(self._nbins, 1)

            sigma_sm_extended = self.sigma_nfw()
            mean_inside_sigma_sm = np.zeros([self._nlens,
                                             original_rbins.shape[0]])

            for i, r in enumerate(original_rbins):
                index_of_rbin = np.where(r_extended == r)[0][0]
                x = r_extended[0:index_of_rbin + 1]
                y = sigma_sm_extended[:, 0:index_of_rbin + 1] * x

                integral = simps(y, x=x, axis=-1, even='first')

                # average of sigma_sm at r < rbin
                mean_inside_sigma_sm[:, i] = (2. / r**2) * integral

            mean_inside_sigma_sm = mean_inside_sigma_sm * (units.Msun /
                                                           units.pc**2)

            # reset original rbins, nbins, x
            self._rbins = original_rbins * units.Mpc
            self._nbins = self._rbins.shape[0]
            _set_dimensionless_radius(self)
            rs_dc_rcrit = self._rs * self._delta_c * self._rho_crit
            self._rs_dc_rcrit = rs_dc_rcrit.reshape(self._nlens,
                                                    1).repeat(self._nbins, 1)
            self._sigma_sm = sigma_sm_rbins  # reset to original sigma_sm

            dsigma_sm = mean_inside_sigma_sm - sigma_sm_rbins

            return dsigma_sm

        if self._sigmaoffset is None:
            finaldeltasigma = _centered_dsigma(self)
        elif np.abs(self._sigmaoffset).sum() == 0:
            finaldeltasigma = _centered_dsigma(self)
        else:
            finaldeltasigma = _offset_dsigma(self)

        return finaldeltasigma


# Notes on Integration
# ------------------------
# Among the choices for numerical integration algorithms, the
# following options were considered:
# (1) scipy.integrate.dblquad is the obvious choice, but is far too
# slow because it makes of order 10^5 calls to the function to be
# integrated, even for generous settings of epsabs, epsrel. Likely
# it is getting stuck in non-smooth portions of the function space.
# (2) scipy.integrate.simps is fast and converges faster than the
# midpoint integration for both the integration over Roff and theta.
# (3) scipy.integrate.romb was somewhat slower than simps and as well
# as the midpoint rule integration.
# (4) midpoint rule integration via a Riemann Sum (the choice used in
# the previous reincarnation of this project in the C programming
# language) is about the same speed as simps, and converges smoothly
# for both integrals, but requires a much larger number of bins to
# converge to the best estimate.
# (5) numpy.trapz underestimates concave down functions.
