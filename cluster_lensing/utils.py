import numpy as np


def check_units_and_type(input, expected_units, num=None, is_scalar=False):
    """Check whether variable has expected units and type.

    If input does not have units and expected units is not None, then the
    output will be assigned those units. If input has units that conflict
    with expected units a ValueError will be raised.

    Parameters
    ----------
    input : array_like or float
        Variable that will be checked for units and type. Variable should
        be 1D or scalar.
    expected_units : astropy.units or None
        Unit expected for input.
    num : int, optional
        Length expected for input, if it is an array or list.
    is_scalar : bool, optional
        Sets whether the input is a scalar quantity. Default is False for
        array_like inputs; set is_scalar=True to check scalar units only.

    Returns
    ----------
    ndarray or float, with astropy.units
        Returns the input array or scalar with expected units, unless a
        conflict of units or array length occurs, which raise errors.
    """
    if hasattr(input, 'unit'):    # check units
        if expected_units is None:
            raise ValueError('Expecting dimensionless input')
        elif input.unit != expected_units:
            raise ValueError('Expecting input units of ' + str(expected_units))
        else:
            dimensionless = input.value

    else:
        dimensionless = input

    # check its a 1D array and/or convert list to array
    if is_scalar is False:
        dimensionfull = check_array_or_list(dimensionless)
    else:
        dimensionfull = dimensionless

    # include units if appropriate
    if expected_units is not None:
        dimensionfull = dimensionfull * expected_units

    # check array length if appropriate
    if num is not None:
        check_input_size(dimensionfull, num)

    return dimensionfull


def check_array_or_list(input):
    """Return 1D ndarray, if input can be converted and elements are
    non-negative."""
    if type(input) != np.ndarray:
        if type(input) == list:
            output = np.array(input)
        else:
            raise TypeError('Expecting input type as ndarray or list.')
    else:
        output = input

    if output.ndim != 1:
        raise ValueError('Input array must have 1 dimension.')

    if np.sum(output < 0.) > 0:
            raise ValueError("Input array values cannot be negative.")

    return output


def check_input_size(array, num):
    """Check that an array has the expected number of elements."""
    if array.shape[0] != num:
        raise ValueError('Input arrays must have the same length (the \
                          number of clusters).')
