import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises

from clusterlensing.clusters import ClusterEnsemble


# ------ TOY DATA FOR TESTING --------------
# Note: some of this depends on cosmology & c(M) relation.
# Assuming defaults of Planck13 and DuttonMaccio, respectively.
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
toy_data_rs = np.array([[0.06898523, 0.06749123],
                        [0.10501577, 0.10030814],
                        [0.42413215, 0.37411303],
                        [0., 0.]])
toy_data_c200 = np.array([[6.52945211, 4.62018029],
                          [5.92732818, 4.29586647],
                          [4.29812433, 3.37326743],
                          [np.inf, np.inf]])
toy_data_deltac = np.array([[16114.78293441, 7270.69406851],
                            [12856.72266431, 6176.07514091],
                            [6183.32001126, 3633.97602181],
                            [np.nan, np.nan]])
# ------------------------------------------


def test_initialization():
    c = ClusterEnsemble(toy_data_z)
    assert_equal(c.z, toy_data_z)
    assert_allclose(c.Dang_l.value, [208.18989, 1697.5794])
    assert_equal(c.number, 2)


def test_initialization_wmap9():
    from astropy.cosmology import WMAP9 as wmap9
    c = ClusterEnsemble(toy_data_z, cosmology=wmap9)
    assert_equal(c.z, toy_data_z)
    assert_allclose(c.Dang_l.value, [203.7027, 1681.5353])
    assert_equal(c.number, 2)


def test_initialization_string_cosmology():
    def use_string_cosmology():
        ClusterEnsemble(toy_data_z, cosmology="WMAP9")
    assert_raises(TypeError, use_string_cosmology)


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
            assert_allclose(toy_data_r200[i] / toy_data_rs[i], c.c200)
        else:
            assert(toy_data_r200[i] / toy_data_rs[i] is not np.real)

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
            assert_allclose(toy_data_r200[i] / toy_data_rs[i], c.c200)
        else:
            assert(toy_data_r200[i] / toy_data_rs[i] is not np.real)

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
    richness = [np.ones(3), np.arange(4), np.arange(5) + 20.]

    def set_n200(val):
        c.n200 = val
    for n in richness:
        assert_raises(ValueError, set_n200, n)


def test_wrong_length_z():
    c = ClusterEnsemble(toy_data_z)
    redshifts = [np.ones(3), np.arange(4), np.arange(5) + 20.]

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
    norm_after = 2. * norm_before
    m_before = c.m200
    c.massrich_norm = norm_after
    m_after = c.m200
    assert_equal(m_before / m_after, np.array([norm_before / norm_after] * 2))


# ------ TOY DATA FOR TESTING NFW ----------

toy_data_rbins = np.array([0.1, 0.26591479, 0.70710678, 1.88030155, 5.])
toy_data_offset = np.array([0.1, 0.1])

toy_data_sigma = np.array([[[6.16761908e+01, 1.39039946e+01, 2.44275736e+00,
                             3.77903918e-01, 5.53525953e-02],
                            [7.96641862e+01, 1.78324931e+01, 3.12034544e+00,
                             4.81847673e-01, 7.05251489e-02]],
                           [[1.27493931e+02, 3.32915500e+01, 6.40080868e+00,
                             1.03341380e+00, 1.54065429e-01],
                            [1.66972278e+02, 4.28673786e+01, 8.15119054e+00,
                             1.30888896e+00, 1.94689165e-01]],
                           [[879.78399678, 391.2004944, 120.01031327,
                             25.88781944, 4.43731976],
                            [1261.43218563, 536.87349277, 156.77363455,
                             32.58239059, 5.47390998]],
                           [np.empty(5) * np.nan, np.empty(5) * np.nan]])

toy_data_deltasigma = np.array([[[65.43997434, 26.39164262, 7.61618077,
                                  1.75644108, 0.35317515],
                                 [85.68440795, 34.26283038, 9.82631838,
                                  2.25736485, 0.45282956]],
                                [[103.99931225, 49.62147716, 16.32117621,
                                  4.0917027, 0.86548029],
                                 [140.17269195, 65.64905033, 21.26478668,
                                  5.27806563, 1.10966763]],
                                [[315.90515388, 246.17865955, 138.86425678,
                                  53.06822795, 14.73246768],
                                 [484.19549909, 364.98607848, 195.93777776,
                                  71.4483272, 19.19583107]],
                                [np.empty(5) * np.nan, np.empty(5) * np.nan]])

toy_data_sigma_off = np.array([[[5.69501008e+01, 1.74504112e+01, 2.52901319,
                                 3.79783870e-01, 5.53770689e-02],
                                [7.38052696e+01, 2.24484132e+01, 3.23098141,
                                 4.84248607e-01, 7.05563626e-02]],
                               [[111.8118779, 39.51432979, 6.60603604,
                                 1.03835891, 0.15413189],
                                [147.0904175, 51.1793199, 8.41578176,
                                 1.31518391, 0.19477342]],
                               [[749.80727064, 407.48783438, 121.93704135,
                                 25.97782737, 4.43885162],
                                [1071.49027696, 563.33246958, 159.53729442,
                                 32.7011807, 5.47586815]],
                               [np.empty(5) * np.nan, np.empty(5) * np.nan]])

toy_data_deltasigma_off = np.array([[[8.36836538, 16.03773137, 7.33259501,
                                      1.72339183, 0.3489577],
                                     [10.91061883, 20.85627022, 9.45937694,
                                      2.21438965, 0.4473347]],
                                    [[14.50275018, 29.23923747, 15.7315571,
                                      4.03049431, 0.8580319],
                                     [19.35639601, 38.80576556, 20.49612042,
                                      5.19710646, 1.09975454]],
                                    [[57.97626837, 136.93183823, 132.08593043,
                                      52.60120424, 14.68947908],
                                     [87.14987636, 203.42283987, 186.76588425,
                                      70.8056412, 19.13516388]],
                                    [np.empty(5) * np.nan,
                                     np.empty(5) * np.nan]])


def test_nfw_centered():
    c = ClusterEnsemble(toy_data_z)

    def _check_sigma(i, j):
        assert_allclose(c.sigma_nfw[j].value, toy_data_sigma[i, j],
                        rtol=1e-4)

    def _check_deltasigma(i, j):
        assert_allclose(c.deltasigma_nfw[j].value, toy_data_deltasigma[i, j],
                        rtol=1e-4)

    for i, n200 in enumerate(toy_data_n200):
        c.n200 = n200
        c.calc_nfw(toy_data_rbins)
        for j in range(c.z.shape[0]):
            yield _check_sigma, i, j
            yield _check_deltasigma, i, j


def test_nfw_offset():
    c = ClusterEnsemble(toy_data_z)

    def _check_sigma(i, j):
        assert_allclose(c.sigma_nfw[j].value, toy_data_sigma_off[i, j],
                        rtol=10**-4)

    def _check_deltasigma(i, j):
        assert_allclose(c.deltasigma_nfw[j].value,
                        toy_data_deltasigma_off[i, j],
                        rtol=10**-4)

    for i, n200 in enumerate(toy_data_n200[:-1]):
        c.n200 = n200
        c.calc_nfw(toy_data_rbins, offsets=toy_data_offset)
        for j in range(c.z.shape[0]):
            yield _check_sigma, i, j
            yield _check_deltasigma, i, j


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
        c.calc_nfw(toy_data_rbins, offsets=toy_data_offset)
        for i in range(c.z.shape[0]):
            yield _check_sigma_off, c.sigma_nfw[i].value
            yield _check_deltasigma_off, c.deltasigma_nfw[i].value
