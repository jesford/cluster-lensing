import numpy as np
from astropy import units

def check_input(input, expected_units):
    #check units
    if hasattr(input, 'unit'):
        if input.unit != expected_units:
            raise ValueError('Expecting input units of ' + str(expected_units))
        else:
            output = input.value
    else:
        output = input
        
    output = check_array_or_list(output) * expected_units
        
    return output

def check_array_or_list(input):
    #check its list or array
    if type(input) != np.ndarray:
        if type(input) == list:
            output = np.array(input)
        else:
            raise TypeError('Expecting input type as ndarray or list.')
    else:
        output = input
    return output


#TO DO: check that array.ndim == 1
