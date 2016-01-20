import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
from astropy import units

from smd_nfw import SurfaceMassDensity


#test units of inputs
ncl = 3
r_s =  np.repeat(0.1,ncl) * units.Mpc
delta_c = np.repeat(10000., ncl) #dimensionless
rho_c = np.repeat(0.2, ncl) * units.Msun / units.pc**2 / units.Mpc
sig_off = np.repeat(0.2,ncl) * units.Mpc
rbinarray = np.logspace(np.log10(0.1), np.log10(5.), num = 10) * units.Mpc

def test_rs_units():
    rs_unitless = r_s.value
    rs_wrongunits = r_s.value * units.Msun

    #test that a dimensionless rs is coverted to Mpc
    smd = SurfaceMassDensity(rs_unitless, delta_c, rho_c,
                             sig_offset = sig_off, rbins = rbinarray)
    assert_equal(smd._rs.unit, units.Mpc)

    #test that a non-Mpc unit on rs raises an error
    assert_raises(ValueError, SurfaceMassDensity, rs_wrongunits, delta_c,
                  rho_c, sig_offset = sig_off, rbins = rbinarray)

def test_rhoc_units():
    rhoc_unitless = rho_c.value
    rhoc_wrongunits = rho_c.value * units.kg/units.m**3

    #test that a dimensionless rho_c is coverted to Msun/Mpc/pc**2
    smd = SurfaceMassDensity(r_s, delta_c, rhoc_unitless,
                             sig_offset = sig_off, rbins = rbinarray)
    assert_equal(smd._rho_crit.unit, units.Msun/units.Mpc/units.pc**2)

    #test that an incorrect unit on rho_c raises an error
    assert_raises(ValueError, SurfaceMassDensity, r_s, delta_c,
                  rhoc_wrongunits, sig_offset = sig_off, rbins = rbinarray)

def test_dc_units():
    dc_wrongunits = delta_c * units.Mpc
    assert_raises(ValueError, SurfaceMassDensity, r_s, dc_wrongunits, rho_c,
                  sig_offset = sig_off, rbins = rbinarray)


#test lists as input
rs_list = list(r_s.value)
dc_list = list(delta_c)
rc_list = list(rho_c.value)
soff_list = list(sig_off.value)
rbin_list = list(rbinarray.value)

def test_list_rs():
    smd = SurfaceMassDensity(rs_list, delta_c, rho_c,
                             sig_offset = sig_off, rbins = rbinarray)
    assert_equal(smd._rs, r_s)

def test_list_dc():
    smd = SurfaceMassDensity(r_s, dc_list, rho_c,
                             sig_offset = sig_off, rbins = rbinarray)
    assert_equal(smd._delta_c, delta_c)

def test_list_rc():
    smd = SurfaceMassDensity(r_s, delta_c, rc_list,
                             sig_offset = sig_off, rbins = rbinarray)
    assert_equal(smd._rho_crit, rho_c)

def test_list_sigoff():
    smd = SurfaceMassDensity(r_s, delta_c, rho_c,
                             sig_offset = soff_list, rbins = rbinarray)
    assert_equal(smd._sigmaoffset, sig_off)

def test_list_rbins():
    smd = SurfaceMassDensity(r_s, delta_c, rho_c,
                             sig_offset = sig_off, rbins = rbin_list)
    assert_equal(smd._rbins, rbinarray)


def test_input_single_values():
    assert_raises(TypeError, SurfaceMassDensity, r_s[0], delta_c, rho_c,
                  sig_offset = sig_off, rbins = rbinarray)
    assert_raises(TypeError, SurfaceMassDensity, r_s, delta_c[0], rho_c,
                  sig_offset = sig_off, rbins = rbinarray)
    assert_raises(TypeError, SurfaceMassDensity, r_s, delta_c, rho_c[0],
                  sig_offset = sig_off, rbins = rbinarray)
    assert_raises(TypeError, SurfaceMassDensity, r_s, delta_c, rho_c,
                  sig_offset = sig_off[0], rbins = rbinarray)
    assert_raises(TypeError, SurfaceMassDensity, r_s, delta_c, rho_c,
                  sig_offset = sig_off, rbins = rbinarray[0])

    
def test_incompatible_lengths():

    assert_raises(ValueError, SurfaceMassDensity, r_s[0:2], delta_c, rho_c,
                  sig_offset = sig_off, rbins = rbinarray)
    
    assert_raises(ValueError, SurfaceMassDensity, r_s, delta_c[0:2], rho_c,
                  sig_offset = sig_off, rbins = rbinarray)

    assert_raises(ValueError, SurfaceMassDensity, r_s, delta_c, rho_c[0:2],
                  sig_offset = sig_off[0:2], rbins = rbinarray)

    assert_raises(ValueError, SurfaceMassDensity, r_s, delta_c, rho_c,
                  sig_offset = sig_off[0:2], rbins = rbinarray)
