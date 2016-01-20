import numpy as np
from astropy import units

def check_units_and_type(input, expected_units):
    #check units
    if hasattr(input, 'unit'):
        if expected_units is None:
            raise ValueError('Expecting dimensionless input')
        elif input.unit != expected_units:
            raise ValueError('Expecting input units of ' + str(expected_units))
        else:
            dimensionless = input.value
        
    else:
        dimensionless = input
        
    dimensionfull = check_array_or_list(dimensionless)

    if expected_units is not None:
        dimensionfull = dimensionfull * expected_units
        
    return dimensionfull


def check_array_or_list(input):
    #check its list or array
    if type(input) != np.ndarray:
        if type(input) == list:
            output = np.array(input)
        else:
            raise TypeError('Expecting input type as ndarray or list.')
    else:
        output = input

    if output.ndim != 1:
        raise ValueError('Input array must have 1 dimension.')
    
    return output


def check_input_size(array, num):
    if array.shape[0] != num:
        raise ValueError('Input arrays must have the same length (the \
                          number of clusters).')


