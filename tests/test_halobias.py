import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.cosmology import Planck13 as cosmo

import halobias

def test_Mnl_z0():
    z = 0.
    h = cosmo.H0.value
    m = (8.73/h) * 10.**12. #non-linear mass
        
    b = halobias.bias(m,z)
    
    assert_allclose(b,0.9236707)#, rtol=1e-04)
