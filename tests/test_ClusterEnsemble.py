import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
from astropy.cosmology import Planck13 as cosmo

from clusters import ClusterEnsemble


#------ TOY DATA FOR TESTING --------------
toy_data_z = np.array([0.05,1.0])
toy_data_n200 = np.array([[10,10] ,[20,20], [200.,200.], [0.,0.]])
toy_data_m200 = np.array([[1.02310868, 1.02310868],
                          [2.7, 2.7],
                          [67.8209337, 67.8209337],
                          [0., 0.]]) * 10**13
toy_data_r200 = np.array([[0.45043573,0.31182166],
                          [0.62246294, 0.43091036],
                          [1.82297271, 1.26198329],
                          [0.,0.]])
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
#------------------------------------------   


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
        c.update_richness(n200)
        yield _check_n_and_m, i
        yield _check_radii, i
        yield _check_c200, i
        yield _check_delta_c, i
        

def test_negative_z():
    redshifts = np.array([[-1.,-999.], [20.,30.,-10.]])
    for z in redshifts:
        assert_raises(ValueError, ClusterEnsemble, z)
        
def test_negative_n200():
    c = ClusterEnsemble(toy_data_z)
    richness = np.array([[-1.,-999.], [30.,-10.]])
    for n in richness:
        assert_raises(ValueError, c.update_richness, n)

def test_wrong_length_richness():
    c = ClusterEnsemble(toy_data_z)
    richness = [np.ones(3), np.arange(4), np.arange(5)+20.]
    for n in richness:
        assert_raises(ValueError, c.update_richness, n)

def test_wrong_length_z():
    c = ClusterEnsemble(toy_data_z)
    redshifts = [np.ones(3), np.arange(4), np.arange(5)+20.]
    for z in redshifts:
        assert_raises(ValueError, c.update_z, z)
    
def test_wrong_length_update_MNrelation():
    c = ClusterEnsemble(toy_data_z)
    assert_raises(TypeError, c.update_massrichrelation, slope=[1.5,2.,2.5])
    assert_raises(TypeError, c.update_massrichrelation, norm=[1.5e14,2.e13,2.5e-2])

def test_update_slope():
    pass
