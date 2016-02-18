import numpy as np
from numpy.testing import assert_allclose, assert_raises

from clusterlensing import cofm


# test data
redshift = [0.1, 0.5, 1.0, 10., 0.]
mass = [1e12, 1e13, 1e14, 1e15]  # solar masses


def _check_c(z, m, answer, relation):
    if relation == 'Duffy':
        c200 = cofm.c_Duffy(z, m)
    elif relation == 'DuttonMaccio':
        c200 = cofm.c_DuttonMaccio(z, m)
    elif relation == 'Prada':
        c200 = cofm.c_Prada(z, m)
    assert_allclose(c200, answer)


def test_Duffy():
    relation = 'Duffy'
    scalar_answers = np.array([[5.97944557, 4.927889,
                                4.06126115, 3.34704011],
                              [5.16835799, 4.25944081,
                               3.51036752, 2.89302767],
                              [4.51472582, 3.72075763,
                               3.06641818, 2.5271521],
                              [2.02610019, 1.66978639,
                               1.37613461, 1.13412498],
                              [6.25338949, 5.15365662,
                               4.24732485, 3.5003823]])

    for z, ans in zip(redshift, scalar_answers):
        _check_c(z, mass, ans, relation)

    arr_arr_ans = np.array([5.97944557, 4.25944081,
                            3.06641818, 1.13412498])

    _check_c(redshift[0:len(mass)], mass, arr_arr_ans, relation)


def test_DuttonMaccio():
    relation = 'DuttonMaccio'
    scalar_answers = np.array([[8.07707505, 6.43952922,
                                5.13397936, 4.09311662],
                              [6.75675873, 5.51745007,
                               4.50545246, 3.67907305],
                              [5.50051123, 4.62810343,
                               3.89406374, 3.27644631],
                              [3.11280793, 4.48902811,
                               6.47369635, 9.33581688],
                              [8.35728502, 6.6231595,
                               5.24886272, 4.15973069]])

    for z, ans in zip(redshift, scalar_answers):
        _check_c(z, mass, ans, relation)

    arr_arr_ans = np.array([8.07707505, 5.51745007,
                            3.89406374, 9.33581688])

    _check_c(redshift[0:len(mass)], mass, arr_arr_ans, relation)


def test_Prada():
    relation = 'Prada'
    # Prada prediction is off the charts for high z...
    scalar_answers = np.array([[6.65883862, 5.71130928,
                                5.06733941, 5.30163572],
                              [5.90404735, 5.23409449,
                               4.93327451, 5.99897362],
                              [5.18782842, 4.84903446,
                               5.08340551, 8.30779577],
                              [7.26222551e+01, 1.77136409e+03,
                               1.23037310e+07, 1.74790223e+20],
                              [6.87424384, 5.85333594,
                               5.12778037, 5.20890574]])

    for z, ans in zip(redshift, scalar_answers):
        _check_c(z, mass, ans, relation)

    arr_arr_ans = np.array([6.65883862, 5.23409449,
                            5.08340551, 1.74790223e+20])

    _check_c(redshift[0:len(mass)], mass, arr_arr_ans, relation)


def test_negative_z():
    dummy_ans = np.zeros(len(mass))
    for relation in ['Duffy', 'DuttonMaccio', 'Prada']:
        assert_raises(ValueError, _check_c, -10, mass, dummy_ans, relation)


def test_negative_m():
    dummy_ans = np.zeros(len(mass))
    for relation in ['Duffy', 'DuttonMaccio', 'Prada']:
        assert_raises(ValueError, _check_c, redshift, -10, dummy_ans, relation)
