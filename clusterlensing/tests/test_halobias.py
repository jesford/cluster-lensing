import numpy as np
from numpy.testing import assert_allclose
from astropy.cosmology import Planck13 as cosmo

from clusterlensing.halobias import bias


def test_Mnl_z0():
    z = 0.
    h = cosmo.h

    def _check_bias_z0(m, ans):
        b = bias(m, z)
        assert_allclose(b, ans)

    # integer multiples of non-linear mass
    M_nl = (8.73 / h) * 10.**12.
    mass = np.array([0., 1., 2., 10., 100.]) * M_nl

    # corresponding bias at z=0
    answer = [0.66, 0.9236707, 1.06577485, 1.64530492, 4.127912607]

    for m, ans in zip(mass, answer):
        yield _check_bias_z0, m, ans
