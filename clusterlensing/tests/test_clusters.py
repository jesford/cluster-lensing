import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises

from clusterlensing.clusters import ClusterEnsemble


# ------ TOY DATA FOR TESTING --------------
# note: some of this depends on cosmology & c(M) relation.
toy_data_z = np.array([0.05, 1.0])
toy_data_n200 = np.array([[10, 10], [20, 20], [200., 200.], [0., 0.]])
toy_data_m200 = np.array([[1.02310868, 1.02310868],
                          [2.7, 2.7],
                          [67.8209337, 67.8209337],
                          [0., 0.]]) * 10**13
toy_data_r200 = np.array([[0.45043573, 0.31182166],
                          [0.62246294, 0.43091036],
                          [1.82297271, 1.26198329],
                          [0., 0.]])
toy_data_rs = np.array([[0.06920826, 0.06765531],
                        [0.10535529, 0.100552],
                        [0.4255034, 0.37502254],
                        [0., 0.]])
toy_data_c200 = np.array([[6.50841, 4.60897531],
                          [5.90822651, 4.28544802],
                          [4.284273, 3.3650865],
                          [np.inf, np.inf]])
toy_data_deltac = np.array([[15993.18343503, 7231.03898592],
                            [12760.73852901, 6142.71245062],
                            [6138.9566454, 3615.01284489],
                            [np.nan, np.nan]])
# ------------------------------------------


def test_initialization():
    c = ClusterEnsemble(toy_data_z)
    assert_equal(c.z, toy_data_z)
    assert_allclose(c.Dang_l.value, [208.18989, 1697.5794])
    assert_equal(c.number, 2)


def test_update_richness():
    c = ClusterEnsemble(toy_data_z)

    def _check_n_and_m(i):
        assert_equal(c.n200, toy_data_n200[i])
        assert_allclose(c.m200.value, toy_data_m200[i])

    def _check_radii(i):
        assert_allclose(c.r200.value, toy_data_r200[i])
        assert_allclose(c.rs.value, toy_data_rs[i])

    def _check_c200(i):
        assert_allclose(toy_data_c200[i], c.c200)
        if c.c200 is np.real:
            assert_allclose(toy_data_r200[i]/toy_data_rs[i], c.c200)
        else:
            assert(toy_data_r200[i]/toy_data_rs[i] is not np.real)

    def _check_delta_c(i):
        assert_allclose(toy_data_deltac[i], c.delta_c)

    for i, n200 in enumerate(toy_data_n200):
        c.n200 = n200
        yield _check_n_and_m, i
        yield _check_radii, i
        yield _check_c200, i
        yield _check_delta_c, i


def test_update_mass():
    c = ClusterEnsemble(toy_data_z)

    def _check_n_and_m(i):
        assert_equal(c.m200.value, toy_data_m200[i])
        assert_allclose(c.n200, toy_data_n200[i])

    def _check_radii(i):
        assert_allclose(c.r200.value, toy_data_r200[i])
        assert_allclose(c.rs.value, toy_data_rs[i])

    def _check_c200(i):
        assert_allclose(toy_data_c200[i], c.c200)
        if c.c200 is np.real:
            assert_allclose(toy_data_r200[i]/toy_data_rs[i], c.c200)
        else:
            assert(toy_data_r200[i]/toy_data_rs[i] is not np.real)

    def _check_delta_c(i):
        assert_allclose(toy_data_deltac[i], c.delta_c)

    for i, m200 in enumerate(toy_data_m200):
        c.m200 = m200
        yield _check_n_and_m, i
        yield _check_radii, i
        yield _check_c200, i
        yield _check_delta_c, i


def test_negative_z():
    redshifts = np.array([[-1., -999.], [20., 30., -10.]])
    for z in redshifts:
        assert_raises(ValueError, ClusterEnsemble, z)


def test_negative_n200():
    c = ClusterEnsemble(toy_data_z)
    richness = np.array([[-1., -999.], [30., -10.]])

    def set_n200(val):
        c.n200 = val
    for n in richness:
        assert_raises(ValueError, set_n200, n)


def test_wrong_length_richness():
    c = ClusterEnsemble(toy_data_z)
    richness = [np.ones(3), np.arange(4), np.arange(5)+20.]

    def set_n200(val):
        c.n200 = val
    for n in richness:
        assert_raises(ValueError, set_n200, n)


def test_wrong_length_z():
    c = ClusterEnsemble(toy_data_z)
    redshifts = [np.ones(3), np.arange(4), np.arange(5)+20.]

    def set_z(val):
        c.z = val
    for z in redshifts:
        assert_raises(ValueError, set_z, z)


def test_wrong_length_update_MNrelation():
    c = ClusterEnsemble(toy_data_z)

    def set_slope(val):
        c.massrich_slope = val

    def set_norm(val):
        c.massrich_norm = val

    assert_raises(TypeError, set_slope, slope=[1.5, 2., 2.5])
    assert_raises(TypeError, set_norm, norm=[1.5e14, 2.e13, 2.5e-2])


def test_update_slope():
    c = ClusterEnsemble(toy_data_z)
    c.n200 = [10, 20]
    slope_before = c._massrich_slope
    slope_after = 2.
    m_before = c.m200
    c.massrich_slope = slope_after
    m_after = c.m200
    assert_equal(m_before[1], m_after[1])
    assert_equal(m_before[0] / m_after[0], (0.5**slope_before) /
                 (0.5**slope_after))


def test_update_norm():
    c = ClusterEnsemble(toy_data_z)
    c.n200 = [10, 20]
    norm_before = c._massrich_norm.value
    norm_after = 2.*norm_before
    m_before = c.m200
    c.massrich_norm = norm_after
    m_after = c.m200
    assert_equal(m_before/m_after, np.array([norm_before/norm_after]*2))


# ------ TOY DATA FOR TESTING NFW ----------

toy_data_rbins = np.array([0.1, 0.26591479, 0.70710678, 1.88030155, 5.])

toy_data_offset = np.array([0.1, 0.1])

# TO DO: eventually python calculation should be ground "truth" (i.e. the 4
# data sets below), as simps is more accurate than midpoint integration.

# below 4 sets were output by c calculation
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

toy_data_sigma_off = np.array([[[5.72922100e+01, 1.74796230e+01, 2.53308800e+00,
                                 3.80508000e-01, 5.54890000e-02],
                                [7.42894330e+01, 2.24888540e+01, 3.23564000e+00,
                                 4.85052000e-01, 7.06790000e-02]],
                               [[112.162248, 39.549804, 6.615806,
                                 1.040329, 0.15445],
                                [147.648409, 51.236081, 8.42712,
                                 1.317355, 0.195118]],
                               [[749.269997, 407.160948, 121.99714,
                                 26.017224, 4.448009],
                                [1071.578332, 563.177646, 159.643017,
                                 32.746632, 5.485526]],
                               [np.empty(5)*np.nan, np.empty(5)*np.nan]])

toy_data_deltasigma_off = np.array([[[8.030923, 20.076035, 7.259893,
                                      1.681247, 0.335841],
                                     [10.466821, 26.085422, 9.367618,
                                      2.160572, 0.430557]],
                                    [[14.09775, 37.534385, 15.599696,
                                      3.928595, 0.824545],
                                     [18.807823, 49.702694, 20.32113,
                                      5.065814, 1.056858]],
                                    [[57.176779, 189.378023, 136.18578,
                                      52.183441, 14.251751],
                                     [85.983442, 280.070251, 191.662768,
                                      70.063303, 18.533542]],
                                    [np.empty(5)*np.nan, np.empty(5)*np.nan]])

# ------------------------------------------
# test c calculations (smd_nfw.c)


def test_nfw_ccalc_centered():
    c = ClusterEnsemble(toy_data_z)

    def _check_sigma(i, j):
        assert_allclose(c.sigma_nfw[j].value, toy_data_sigma[i, j])

    def _check_deltasigma(i, j):
        assert_allclose(c.deltasigma_nfw[j].value, toy_data_deltasigma[i, j])

    for i, n200 in enumerate(toy_data_n200):
        c.n200 = n200
        c.calc_nfw(toy_data_rbins, use_c=True)
        for j in range(c.z.shape[0]):
            yield _check_sigma, i, j
            yield _check_deltasigma, i, j


def test_nfw_ccalc_offset():
    c = ClusterEnsemble(toy_data_z)

    def _check_sigma(i, j):
        assert_allclose(c.sigma_nfw[j].value, toy_data_sigma_off[i, j])

    def _check_deltasigma(i, j):
        assert_allclose(c.deltasigma_nfw[j].value,
                        toy_data_deltasigma_off[i, j])

    for i, n200 in enumerate(toy_data_n200):
        c.n200 = n200
        c.calc_nfw(toy_data_rbins, offsets=toy_data_offset, use_c=True)
        for j in range(c.z.shape[0]):
            yield _check_sigma, i, j
            yield _check_deltasigma, i, j


# ------------------------------------------
# test python calculations

def test_nfw_centered():
    c = ClusterEnsemble(toy_data_z)

    def _check_sigma(i, j):
        assert_allclose(c.sigma_nfw[j].value, toy_data_sigma[i, j],
                        rtol=10**-5)

    def _check_deltasigma(i, j):
        assert_allclose(c.deltasigma_nfw[j].value, toy_data_deltasigma[i, j],
                        rtol=10**-5)

    for i, n200 in enumerate(toy_data_n200):
        c.n200 = n200
        c.calc_nfw(toy_data_rbins)
        for j in range(c.z.shape[0]):
            yield _check_sigma, i, j
            yield _check_deltasigma, i, j


def test_nfw_offset():
    c = ClusterEnsemble(toy_data_z)

    def _check_sigma(i, j):
        # tolerance is poor because I'm comparing the midpoint integration in c
        # to the simps integration in python (should be much better).
        assert_allclose(c.sigma_nfw[j].value, toy_data_sigma_off[i, j],
                        rtol=10**-1)

    #def _check_deltasigma(i,j):
    #    assert_allclose(c.deltasigma_nfw[j].value,
    #                    toy_data_deltasigma_off[i,j],
    #                    rtol=10**-1)

    for i, n200 in enumerate(toy_data_n200[:-1]):
        c.n200 = n200
        c.calc_nfw(toy_data_rbins, offsets=toy_data_offset)
        for j in range(c.z.shape[0]):
            yield _check_sigma, i, j
            #yield _check_deltasigma, i, j


# ------------------------------------------


def test_for_infs_in_miscentered_c_calc():
    c = ClusterEnsemble(toy_data_z)

    def _check_sigma_off(arr):
        if np.isnan(arr.sum()):
            raise ValueError('sigma_off result contains NaN', arr)
        if np.isinf(arr.sum()):
            raise ValueError('sigma_off result contains Inf', arr)

    def _check_deltasigma_off(arr):
        if np.isnan(arr.sum()):
            raise ValueError('sigma_off result contains NaN', arr)
        if np.isinf(arr.sum()):
            raise ValueError('sigma_off result contains Inf', arr)

    # last element in toy_data is n200=0 -> NaN (skip for this check)
    for n200 in toy_data_n200[:-1]:
        c.n200 = n200
        c.calc_nfw(toy_data_rbins, offsets=toy_data_offset, use_c=True)
        for i in range(c.z.shape[0]):
            yield _check_sigma_off, c.sigma_nfw[i].value
            yield _check_deltasigma_off, c.deltasigma_nfw[i].value
