import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
from astropy import units

from clusterlensing.nfw import SurfaceMassDensity


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
    rhoc_wrongunits = rho_c.value * units.kg / units.m**3

    # test that a dimensionless rho_c is coverted to Msun/Mpc/pc**2
    smd = SurfaceMassDensity(r_s, delta_c, rhoc_unitless,
                             offsets=sig_off, rbins=rbinarray)
    assert_equal(smd._rho_crit.unit, units.Msun / units.Mpc / units.pc**2)

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
# test NFW centered profiles

toy_data_rbins = np.array([0.1, 0.26591479, 0.70710678,
                           1.88030155, 5.]) * units.Mpc
toy_data_z = np.array([0.05, 1.0])
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
                          [0.13363, 0.4028]]) * (units.Msun / units.Mpc /
                                                 (units.pc**2))

toy_data_sigma = np.array([[[61.676628, 13.91868, 2.4468037,
                             0.37863308, 0.055465539],
                            [79.68557, 17.851227, 3.1250101,
                             0.48266501, 0.070650539]],
                           [[127.43513, 33.316155, 6.4106052,
                             1.035399, 0.15438678],
                            [166.96108, 42.902785, 8.1626417,
                             1.3110978, 0.19504083]],
                           [[878.18174, 390.91924, 120.07696,
                             25.927776, 4.446575],
                            [1260.0668, 536.75538, 156.88796,
                             32.628947, 5.4837611]],
                           [np.empty(5) * np.nan, np.empty(5) * np.nan]])

toy_data_deltasigma = np.array([[[65.309133, 26.372068, 7.6175566,
                                  1.7577702, 0.35356601],
                                 [85.578152, 34.252649, 9.8301535,
                                  2.2592132, 0.45331823]],
                                [[103.74144, 49.563266, 16.319755,
                                  4.0942593, 0.8663956],
                                 [139.95018, 65.609361, 21.269033,
                                  5.2818827, 1.1108229]],
                                [[314.79661, 245.51579, 138.66486,
                                  53.056979, 14.742129],
                                 [483.0392, 364.3582, 195.7899,
                                  71.458238, 19.210324]],
                                [np.empty(5) * np.nan, np.empty(5) * np.nan]])


def test_centered_profiles():

    def _check_sigma(i):
        assert_allclose(sigma_py.value, toy_data_sigma[i],
                        rtol=1e-04)

    def _check_deltasigma(i):
        assert_allclose(deltasigma_py.value, toy_data_deltasigma[i],
                        rtol=1e-04)
        # note: default tolerance is (rtol=1e-07, atol=0)

    zipped_inputs = zip(toy_data_rs, toy_data_dc, toy_data_rhoc)

    # check all 4 sets of toy_data:
    for i, (r_s, delta_c, rho_c) in enumerate(zipped_inputs):

        smd = SurfaceMassDensity(r_s, delta_c, rho_c, rbins=toy_data_rbins)
        sigma_py = smd.sigma_nfw()
        _check_sigma(i)

        deltasigma_py = smd.deltasigma_nfw()
        _check_deltasigma(i)


# ----------------------------
# test NFW offset profiles

toy_data_offset = np.array([0.1, 0.1]) * units.Mpc

toy_data_sigma_off = np.array([[[56.922836, 17.461102, 2.5331495,
                                 0.3805162, 0.055490059],
                                [73.797572, 22.464487, 3.2357614,
                                 0.48506959, 0.070681805]],
                               [[111.72633, 39.527318, 6.6159646,
                                 1.0403518, 0.15445337],
                                [147.04471, 51.205603, 8.427435,
                                 1.3174017, 0.19512522]],
                               [[748.5117, 407.12286, 121.99994,
                                 26.017806, 4.4481085],
                                [1070.3932, 563.12414, 159.64883,
                                 32.747803, 5.4857215]],
                               [np.empty(5) * np.nan, np.empty(5) * np.nan]])

toy_data_deltasigma_off = np.array([[[8.3569013, 16.021837, 7.3340382,
                                      1.7247532, 0.34935389],
                                     [10.902221, 20.846143, 9.4631887,
                                      2.2162585, 0.44782718]],
                                    [[14.476809, 29.198514, 15.730196,
                                      4.0331203, 0.85895897],
                                     [19.335676, 38.775763, 20.500251,
                                      5.2009726, 1.1009191]],
                                    [[57.800595, 136.55704, 131.88898,
                                      52.590289, 14.699197],
                                     [86.975707, 203.06399, 186.61751,
                                      70.815764, 19.149708]],
                                    [np.empty(5) * np.nan,
                                     np.empty(5) * np.nan]])


def test_miscentered_profiles():

    def _check_sigma(i):
        assert_allclose(sigma_py_off.value, toy_data_sigma_off[i],
                        rtol=1e-04)

    def _check_deltasigma(i):
        assert_allclose(deltasigma_py_off.value, toy_data_deltasigma_off[i],
                        rtol=1e-04)

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
