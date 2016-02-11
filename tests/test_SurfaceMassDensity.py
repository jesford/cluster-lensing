import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
from astropy import units

from smd_nfw import SurfaceMassDensity


# ----------------------------
# test midpoint integration

from smd_nfw import midpoint


def test_midpoint_linear():
    x = np.linspace(0, 10, 10)
    y = x
    integral = midpoint(y, x=x)
    assert_allclose(integral, 50.)

    dx = float(x[1]-x[0])
    print('dx', dx)
    integral2 = midpoint(y, dx=dx)
    assert_allclose(integral2, 50.)


def test_midpoint_linear_neg():
    x = np.linspace(4, 10, 6)
    y = -1.*x
    integral = midpoint(y, x=x)
    assert_allclose(integral, -42.)


def test_midpoint_quadratic():
    x = np.linspace(0, 10, 100)
    y = x**2
    integral = midpoint(y, x=x)
    # exact integral would be 333.3333...
    # but midpoint should yield approx 333.3503
    assert_allclose(integral, 333.35, atol=0.01)


# ----------------------------
# test units of inputs

ncl = 3
r_s = np.repeat(0.1, ncl) * units.Mpc
delta_c = np.repeat(10000., ncl)  # dimensionless
rho_c = np.repeat(0.2, ncl) * units.Msun / units.pc**2 / units.Mpc
sig_off = np.repeat(0.2, ncl) * units.Mpc
rbinarray = np.logspace(np.log10(0.1), np.log10(5.), num=10) * units.Mpc


def test_rs_units():
    rs_unitless = r_s.value
    rs_wrongunits = r_s.value * units.Msun

    # test that a dimensionless rs is coverted to Mpc
    smd = SurfaceMassDensity(rs_unitless, delta_c, rho_c,
                             offsets=sig_off, rbins=rbinarray)
    assert_equal(smd._rs.unit, units.Mpc)

    # test that a non-Mpc unit on rs raises an error
    assert_raises(ValueError, SurfaceMassDensity, rs_wrongunits, delta_c,
                  rho_c, offsets=sig_off, rbins=rbinarray)


def test_rhoc_units():
    rhoc_unitless = rho_c.value
    rhoc_wrongunits = rho_c.value * units.kg/units.m**3

    # test that a dimensionless rho_c is coverted to Msun/Mpc/pc**2
    smd = SurfaceMassDensity(r_s, delta_c, rhoc_unitless,
                             offsets=sig_off, rbins=rbinarray)
    assert_equal(smd._rho_crit.unit, units.Msun/units.Mpc/units.pc**2)

    # test that an incorrect unit on rho_c raises an error
    assert_raises(ValueError, SurfaceMassDensity, r_s, delta_c,
                  rhoc_wrongunits, offsets=sig_off, rbins=rbinarray)


def test_dc_units():
    dc_wrongunits = delta_c * units.Mpc
    assert_raises(ValueError, SurfaceMassDensity, r_s, dc_wrongunits, rho_c,
                  offsets=sig_off, rbins=rbinarray)


# ----------------------------
# test lists as input

rs_list = list(r_s.value)
dc_list = list(delta_c)
rc_list = list(rho_c.value)
soff_list = list(sig_off.value)
rbin_list = list(rbinarray.value)


def test_list_rs():
    smd = SurfaceMassDensity(rs_list, delta_c, rho_c,
                             offsets=sig_off, rbins=rbinarray)
    assert_equal(smd._rs, r_s)


def test_list_dc():
    smd = SurfaceMassDensity(r_s, dc_list, rho_c,
                             offsets=sig_off, rbins=rbinarray)
    assert_equal(smd._delta_c, delta_c)


def test_list_rc():
    smd = SurfaceMassDensity(r_s, delta_c, rc_list,
                             offsets=sig_off, rbins=rbinarray)
    assert_equal(smd._rho_crit, rho_c)


def test_list_sigoff():
    smd = SurfaceMassDensity(r_s, delta_c, rho_c,
                             offsets=soff_list, rbins=rbinarray)
    assert_equal(smd._sigmaoffset, sig_off)


def test_list_rbins():
    smd = SurfaceMassDensity(r_s, delta_c, rho_c,
                             offsets=sig_off, rbins=rbin_list)
    assert_equal(smd._rbins, rbinarray)


def test_input_single_values():
    assert_raises(TypeError, SurfaceMassDensity, r_s[0], delta_c, rho_c,
                  offsets=sig_off, rbins=rbinarray)
    assert_raises(TypeError, SurfaceMassDensity, r_s, delta_c[0], rho_c,
                  offsets=sig_off, rbins=rbinarray)
    assert_raises(TypeError, SurfaceMassDensity, r_s, delta_c, rho_c[0],
                  offsets=sig_off, rbins=rbinarray)
    assert_raises(TypeError, SurfaceMassDensity, r_s, delta_c, rho_c,
                  offsets=sig_off[0], rbins=rbinarray)
    assert_raises(TypeError, SurfaceMassDensity, r_s, delta_c, rho_c,
                  offsets=sig_off, rbins=rbinarray[0])


def test_incompatible_lengths():

    assert_raises(ValueError, SurfaceMassDensity, r_s[0:2], delta_c, rho_c,
                  offsets=sig_off, rbins=rbinarray)

    assert_raises(ValueError, SurfaceMassDensity, r_s, delta_c[0:2], rho_c,
                  offsets=sig_off, rbins=rbinarray)

    assert_raises(ValueError, SurfaceMassDensity, r_s, delta_c, rho_c[0:2],
                  offsets=sig_off[0:2], rbins=rbinarray)

    assert_raises(ValueError, SurfaceMassDensity, r_s, delta_c, rho_c,
                  offsets=sig_off[0:2], rbins=rbinarray)


# ----------------------------
# test NFW profiles

toy_data_rbins = np.array([0.1, 0.26591479, 0.70710678,
                           1.88030155, 5.]) * units.Mpc
toy_data_z = np.array([0.05, 1.0])
#toy_data_n200 = np.array([[10,10] ,[20,20], [200.,200.], [0.,0.]])
toy_data_rs = np.array([[0.06920826, 0.06765531],
                        [0.10535529, 0.100552],
                        [0.4255034, 0.37502254],
                        [0., 0.]]) * units.Mpc
toy_data_dc = np.array([[15993.18343503, 7231.03898592],
                        [12760.73852901, 6142.71245062],
                         [6138.9566454, 3615.01284489],
                         [np.nan, np.nan]])
toy_data_rhoc = np.array([[0.13363, 0.4028],
                          [0.13363, 0.4028],
                          [0.13363, 0.4028],
                          [0.13363, 0.4028]]) * \
                          units.Msun/units.Mpc/(units.pc**2)

toy_data_sigma = np.array([[[6.16767240e+01, 1.39187020e+01, 2.44680800e+00,
                            3.78634000e-01, 5.54660000e-02],
                            [7.96846330e+01, 1.78510170e+01, 3.12497300e+00,
                             4.82659000e-01, 7.06500000e-02]],
                            [[1.27435338e+02, 3.33162080e+01, 6.41061600e+00,
                              1.03540100e+00, 1.54387000e-01],
                              [1.66959114e+02, 4.29022790e+01, 8.16254600e+00,
                               1.31108200e+00, 1.95039000e-01]],
                            [[878.183118, 390.919853, 120.077149,
                              25.927817, 4.446582], [1260.051922, 536.749041,
                            156.886111, 32.628563, 5.483696]],
                            [np.empty(5)*np.nan, np.empty(5)*np.nan]])

toy_data_deltasigma = np.array([[[65.309234, 26.372108, 7.617568,
                                  1.757773, 0.353567],
                                  [85.577146, 34.252247, 9.830038,
                                   2.259187, 0.453313]],
                                [[103.741612, 49.563345, 16.319782,
                                  4.094266, 0.866397],
                                [139.948531, 65.608588, 21.268782,
                                 5.281821, 1.11081]],
                                 [[314.797102, 245.516172, 138.665076,
                                   53.057064, 14.742152],
                                 [483.0335, 364.353901, 195.78759,
                                  71.457397, 19.210098]],
                                [np.empty(5)*np.nan, np.empty(5)*np.nan]])

toy_data_offset = np.array([0.1, 0.1]) * units.Mpc

toy_data_sigma_off = np.array([[[5.68899837e+01, 1.74622593e+01, 2.53432922e+00,
                                 3.80650177e-01, 5.55088889e-02],
                                 [7.37526890e+01, 2.24655649e+01, 3.23722744e+00,
                                  4.85233907e-01, 7.07048456e-02]],
                               [[111.6894906, 39.53246892, 6.6189454,
                                 1.04071767, 0.15450579],
                                [146.99110648, 51.21132051, 8.43113343,
                                 1.31784724, 0.19518881]],
                               [[748.51062289, 407.13399209, 122.04716135,
                                  26.02685341, 4.44961635],
                                [1070.35938187, 563.11603663, 159.70937527,
                                 32.75876948, 5.48750814]],
                               [[np.nan, np.nan, np.nan, np.nan, np.nan],
                                [np.nan, np.nan, np.nan, np.nan, np.nan]]])

#toy_data_deltasigma_off = #need to implement this calculation in python still


def test_centered_profiles():

    def _check_sigma(i):
        assert_allclose(sigma_py.value, toy_data_sigma[i],
                        rtol=1e-04, atol=0)

    def _check_deltasigma(i):
        assert_allclose(deltasigma_py.value, toy_data_deltasigma[i],
                        rtol=1e-04, atol=0)
        # note: default tolerance is (rtol=1e-07, atol=0)

    zipped_inputs = zip(toy_data_rs, toy_data_dc, toy_data_rhoc)

    # check all 4 sets of toy_data:
    for i, (r_s, delta_c, rho_c) in enumerate(zipped_inputs):

        smd = SurfaceMassDensity(r_s, delta_c, rho_c, rbins=toy_data_rbins)
        sigma_py = smd.sigma_nfw()
        _check_sigma(i)

        deltasigma_py = smd.deltasigma_nfw()
        _check_deltasigma(i)


# this test is slow!
# leading underscore means nosetest won't run it...
def _test_miscentered_profiles():

    def _check_sigma(i):
        assert_allclose(sigma_py_off.value, toy_data_sigma_off[i],
                        rtol=1e-04, atol=0)

    def _check_deltasigma(i):
        assert_allclose(deltasigma_py_off.value, toy_data_deltasigma_off[i],
                        rtol=1e-04, atol=0)
        # note: default tolerance is (rtol=1e-07, atol=0)

    zipped_inputs = zip(toy_data_rs, toy_data_dc, toy_data_rhoc)

    # check all 4 sets of toy_data:
    for i, (r_s, delta_c, rho_c) in enumerate(zipped_inputs):

        smd = SurfaceMassDensity(r_s, delta_c, rho_c,
                                 offsets=toy_data_offset,
                                 rbins=toy_data_rbins)
        sigma_py_off = smd.sigma_nfw()
        _check_sigma(i)

        deltasigma_py_off = smd.deltasigma_nfw()
        _check_deltasigma(i)
