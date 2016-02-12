"""Add the module level docstring..."""

from __future__ import absolute_import, division, print_function

import numpy as np
from astropy import units
from scipy.interpolate import interp1d
from scipy.integrate import simps
#romb, cumtrapz
#quad, dblquad

import utils


def midpoint(y, x=None, dx=1., axis=-1):
    """Integrate using the midpoint rule.

    Parameters
    ----------
    y : ndarray
        Array to be integrated.
    x : ndarray, optional
        If given, points at which `y` is sampled.
    dx : float, optional
        Spacing of integration points along axis of `y`. Only used when
        `x` is None. Default is 1.
    axis : int, optional
        Axis along which to integrate. Default is the last axis.

    Returns
    ----------
    ndarray
        Result of integrating along an axis of y.

    Notes
    ----------
    If x is given, its first and last value are the start and end of the
    integration region (as opposed to the midpoints of integration bins).
    """
    if x is None:
        if type(dx) != float:
            raise ValueError('type(dx) must be float.')
        else:
            # length of dx_array should be number of y intervals
            dx_array = np.ones(y.shape[axis]-1)*dx
    elif x.shape[0] != y.shape[axis]:
        raise ValueError('x and y have incompatible shapes.')
    else:
        dx_array = x[1:] - x[:-1]

    try:
        xm = (x[1:] + x[:-1]) / 2.
    except TypeError:
        x = np.arange(0, y.shape[axis]) * dx
        xm = (x[1:] + x[:-1]) / 2.

    f = interp1d(x, y, axis=axis)
    ym = f(xm)

    yshape = ym.shape
    xshape = [1] * len(yshape)
    xshape[axis] = yshape[axis]
    dx_array.shape = tuple(xshape)

    integral = (ym * dx_array).sum(axis=axis)

    return integral


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
        x = radii_4D/rs_4D

    else:
        # 1D array of radii and clusters, reshape & broadcast
        radii_repeated = radii.reshape(1, self._nbins)
        rs_repeated = self._rs.reshape(self._nlens, 1)
        x = radii_repeated/rs_repeated

    x = x.value
    if 0. in x:
        x[np.where(x == 0.)] = 1.e-10  # hack to avoid infs in sigma

    # dimensionless radius
    self._x = x

    # set the 3 cases of dimensionless radius x
    self._x_small = np.where(self._x < 1.-1.e-6)
    self._x_big = np.where(self._x > 1.+1.e-6)
    self._x_one = np.where(np.abs(self._x-1) <= 1.e-6)


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

    Methods
    ----------
    sigma_nfw()
        Calculate surface mass density Sigma.
    deltasigma_nfw()
        Calculate differential surface mass density DeltaSigma.

    See Also
    ----------
    ClusterEnsemble : parameters and profiles for a sample of clusters.
        This class provides an interface to SurfaceMassDensity, and tracks
        a DataFrame of parameters as well as nfw profiles for many clusters
        at once, only requiring the user to specify cluster z and richness,
        at a minimum.
    """
    def __init__(self, rs, delta_c, rho_crit, offsets=None, rbins=None):
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

        self._nbins = self._rbins.shape[0]
        self._nlens = self._rs.shape[0]

        if offsets is not None:
            self._sigmaoffset = utils.check_units_and_type(offsets, units.Mpc)
            utils.check_input_size(self._sigmaoffset, self._nlens)
        else:
            self._sigmaoffset = offsets  #None

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

        Notes
        ----------
        Among the choices for numerical integration algorithms, the
        following were considered:
        (1) scipy.integrate.dblquad is the obvious choice, but is far too
        slow because it makes of order 10^5 calls to the function to be
        integrated, even for generous settings of epsabs, epsrel. Likely
        it is getting stuck in non-smooth portions of the function space.
        (2) scipy.integrate.simps is fast and converges faster than the
        midpoint integration for both the integration over Roff and theta. 
        (3) scipy.integrate.romb was somewhat slower than simps and as well
        as the midpoint rule integration.
        (4) midpoint rule integration via a Riemann Sum was about the same
        speed as simps, and converges smoothly for both integrals, but
        requires a larger number of bins to converge to the best estimate.
        (5) numpy.trapz underestimates concave down functions.
        """
        def _centered_sigma(self):
            # perfectly centered cluster case

            # calculate f
            bigF = np.zeros_like(self._x)
            f = np.zeros_like(self._x)

            numerator_arg = ((1. / self._x[self._x_small]) +
                             np.sqrt((1. / (self._x[self._x_small]**2)) - 1.))
            denominator = np.sqrt(1. - (self._x[self._x_small]**2))
            bigF[self._x_small] = np.log(numerator_arg)/denominator

            bigF[self._x_big] = (np.arccos(1./self._x[self._x_big]) /
                                 np.sqrt(self._x[self._x_big]**2 - 1.))

            f = (1. - bigF) / (self._x**2 - 1.)
            f[self._x_one] = 1./3.
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
            numRoff = 200
            numTh = 200  # TO DO: option for user to set this
            #print('numRoff, numTh:', numRoff, numTh)

            numRbins = self._nbins
            maxsig = self._sigmaoffset.value.max()
            # inner/outer bin edges (do NOT set endpoint=False!)
            roff_1D = np.linspace(0., 4.*maxsig, numRoff)
            theta_1D = np.linspace(0., 2.*np.pi, numTh)
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
            inner_integrand = sigma.value/(2.*np.pi)

            # ----- integrate over theta axis -----
            # NOTE: simps converges much faster than midpoint (both do
            # so smoothly). ~200 theta bins is good for simps.

            sigma_of_RgivenRoff = simps(inner_integrand, x=theta_1D, axis=0,
                                        even='first')
            #sigma_of_RgivenRoff = midpoint(inner_integrand, x=theta_1D, axis=0)
            # -------------------------------------

            # theta is gone, now dimensions are: (numRoff,numRbins,nlens)
            sig_off_3D = self._sigmaoffset.value.reshape(1, 1, self._nlens)
            roff_v2 = roff_1D.reshape(numRoff, 1, 1)
            PofRoff = (roff_v2/(sig_off_3D**2) *
                       np.exp(-0.5*(roff_v2 / sig_off_3D)**2))

            dbl_integrand = sigma_of_RgivenRoff * PofRoff

            # ----- integrate over Roff axis -----
            # NOTE: simps oscillates around final value, while midpoint
            # underestimates and rises smoothly to final value.
            # for simps ~200 roff bins is good.
            # (integration axis=0 after theta is gone).

            sigma_smoothed = simps(dbl_integrand, x=roff_1D, axis=0,
                                   even='first')
            #sigma_smoothed = midpoint(dbl_integrand, x=roff_1D, axis=0)
            # -------------------------------------

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

    def deltasigma_nfw(self, mp=False):
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

            both_3a = (4. / (self._x**2)) * np.log(self._x/2.)
            both_3b = 2. / (self._x**2-1.)
            g = firstpart * secondpart + both_3a - both_3b

            g[self._x_one] = (10./3.) + 4.*np.log(0.5)

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

            # get offset sigma inside min(rbins)
            # ...could be sampled more finely both at r < and > min(rbins)...
            numR_inner = 20  # this really affects speed!

            # METHOD 1: this r_extended is identical to c's Rp
            #dR_inner = original_rbins.min()/numR_inner
            #r_inner = np.arange(0.5*dR_inner, original_rbins.min(), dR_inner)
            #r_midpoints = 0.5 * (original_rbins[:-1] + original_rbins[1:])
            #r_extended = np.hstack([r_inner, r_midpoints])

            # METHOD 2: gives similar output to c for offset deltasigma
            # use outer edges of rbins, starting at ~0->max(rbins)
            r_inner = np.linspace(1.e-08, original_rbins.min(),
                                  numR_inner, endpoint=False)
            # want to integrate from 0->min(rbins) for delta_sigma(min(rbins))
            r_extended = np.hstack([r_inner, original_rbins])

            # TO DO?
            # METHOD 3: try using log bins across full range 0->max(rbins)
            # for sigma_sm_extended, instead of inner + midpoints
            #prec = 50
            #r_ext_log = np.logspace(np.log10(1.e-10),
            #                        np.log10(np.max(original_rbins)),
            #                        num = prec)
            #r_extended = r_ext_log

            #print('r_extended\n', r_extended)

            # set temporary extended rbins, nbins, x, rs_dc_rcrit array
            self._rbins = r_extended * units.Mpc
            self._nbins = self._rbins.shape[0]
            _set_dimensionless_radius(self)  # uses _rbins, _nlens
            rs_dc_rcrit = self._rs * self._delta_c * self._rho_crit
            self._rs_dc_rcrit = rs_dc_rcrit.reshape(self._nlens,
                                                    1).repeat(self._nbins, 1)

            sigma_sm_extended = self.sigma_nfw()

            # for Method 1, this is pretty close to c's sigma_smoothed_Rp
            # (it is identical to 3rd-4th digit)
            #print('sigma_sm_extended[0,:]\n', sigma_sm_extended[0,:])

            mean_inside_sigma_sm = np.zeros([self._nlens,
                                             original_rbins.shape[0]])

            for i, r in enumerate(original_rbins):
                x = r_extended[0:(i+numR_inner+1)]
                y = sigma_sm_extended[:, 0:(i+numR_inner+1)] * x

                if mp is True:
                    integral = midpoint(y, x=x, axis=-1)
                else:
                    integral = simps(y, x=x, axis=-1, even='first')

                # average of sigma_sm at r < rbin
                mean_inside_sigma_sm[:, i] = (2. / r**2) * integral

            mean_inside_sigma_sm = mean_inside_sigma_sm * (units.Msun /
                                                           units.pc**2)

            # for Method 2, this is pretty close! (identical to 2nd-3rd digit)
            # for Method 1, this is wildy off (factors of a few)
            #print('mean_inside_sigma_sm[0,:]', mean_inside_sigma_sm[0,:])

            #print('sigma_sm_rbins[0,:]', sigma_sm_rbins[0,:])

            # reset original rbins, nbins, x
            self._rbins = original_rbins * units.Mpc
            self._nbins = self._rbins.shape[0]
            _set_dimensionless_radius(self)
            rs_dc_rcrit = self._rs * self._delta_c * self._rho_crit
            self._rs_dc_rcrit = rs_dc_rcrit.reshape(self._nlens,
                                                    1).repeat(self._nbins, 1)
            self._sigma_sm = sigma_sm_rbins  # reset to original rbins sigma_sm

            dsigma_sm = mean_inside_sigma_sm - sigma_sm_rbins

            return dsigma_sm

        if self._sigmaoffset is None:
            finaldeltasigma = _centered_dsigma(self)
        elif np.abs(self._sigmaoffset).sum() == 0:
            finaldeltasigma = _centered_dsigma(self)
        else:
            finaldeltasigma = _offset_dsigma(self)

        return finaldeltasigma
