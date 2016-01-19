import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
from astropy import units

from smd_nfw import SurfaceMassDensity


def test_inputs():
    r_s =  0.1 * units.Mpc
    delta_c = 10000. #dimensionless
    rho_c = 0.2 * units.Msun / units.pc**2 / units.Mpc

    rbinarray = np.logspace(np.log10(0.1), np.log10(5.), num = 10) * units.Mpc
    sig_off = 0.2 * units.Mpc

    def _test_single_values():
        smd = SurfaceMassDensity(r_s, delta_c, rho_c,
                                 sig_offset = sig_off, rbins = rbinarray[0])
        assert_equal(smd._rs, r_s)
        assert_equal(smd._deltac, delta_c)
        assert_equal(smd._rho_crit, rho_c)
        #assert_equal(smd._sigma

    #def _test_lists():
    #    pass

    #def _test_arrays_unitless():
    #    pass

    #def _test_wrongunits():
    #    pass
    

#assert_raises(TypeError, SurfaceMassDensity(r_s.value, delta_c, rho_c,
#                                                    sig_offset = sig_off,
#                                                    rbins = rbinarray))
    

#def test_initialization():
#    rs = 0.1 #Mpc
    
