"""
Programmed by: Johan Esteban Cuervo Chica

This module includes color space transformations
"""

import numpy as np

def invert_f(arg_x: np.ndarray | float) -> np.ndarray | float:
    """
    Function invert f for convert lab to xyz
    supports float values or list of values

    Args:
        arg_x (list): values

    Returns:
        list: values
    """
    delta = 6/29

    try:
        arg_x = float(arg_x)

        if arg_x > delta:
            return arg_x**3

        else:

            return 3*(delta**2)*(arg_x-4/29)

    except TypeError:
        pass

    result = np.zeros_like(arg_x)

    max_index = np.where(arg_x > delta)
    min_index = np.where(arg_x <= delta)

    result[max_index] = np.power(arg_x[max_index], 3)
    result[min_index] = 3*(delta**2)*(arg_x[min_index] - 4/29)

    return result

def invert_g(arg_x: np.ndarray | float) -> np.ndarray | float:
    """
    Function invert f for convert lab to xyz
    supports float values or list of values

    Args:
        arg_x (list): values

    Returns:
        list: values
    """
    delta = 0.0031308
    alpha = 0.055
    gamma = 2.4

    try:
        arg_x = float(arg_x)
        if arg_x > 1:
            arg_x = 1

        if arg_x > delta:
            return (1 + alpha)*np.power(arg_x,1/gamma) - alpha

        else:

            return 12.92*arg_x

    except TypeError:
        pass

    result = np.zeros_like(arg_x)

    arg_x[np.where(arg_x>1)] = 1
    max_index = np.where(arg_x > delta)
    min_index = np.where(arg_x <= delta)

    result[max_index] = (1+alpha)*np.power(arg_x[max_index], 1/gamma) - alpha
    result[min_index] = 12.92*arg_x[min_index]

    return result

def lab2xyz(values: np.ndarray, illuminant: str = 'D65') -> np.ndarray:
    """
    Conversión to values Cie Lab to XYZ

    Args:
        values (np.ndarray): Values to transform
        illuminant (str, optional): standard illuminant to use['D65','D50']. Defaults to 'D65'.

    Returns:
        np.ndarray: transformed values
    """

    if illuminant.upper() == 'D65':
        xyzn = [95.0489, 100, 108.8840]

    elif illuminant.upper() == 'D50':
        xyzn = [96.4212, 100, 82.5188]

    else:
        raise ValueError(f"Illuminant incorrect options: 'D65','D50' joined: {illuminant}")

    values = np.array(values)
    origin_shape = np.shape(values)

    table_val = values.reshape((-1,3))

    result = np.zeros_like(table_val)

    result[:,0] = (table_val[:,0] + 16)/ 116 + table_val[:,1]/500
    result[:,1] = (table_val[:,0] + 16)/ 116
    result[:,2] = (table_val[:,0] + 16)/ 116 - table_val[:,2]/200

    result = invert_f(result)

    result*=xyzn

    return result.reshape(origin_shape)/100

def xyz2rgb(values: np.ndarray) -> np.ndarray:
    """
    Conversión to values Cie XYZ to sRGB Values

    Args:
        values (np.ndarray): Values to transform

    Returns:
        np.ndarray: transformed values. Matrix uint8
    """

    matrix = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ])

    values = np.array(values)
    origin_shape = np.shape(values)

    table_val = values.reshape((-1,3))

    result = (matrix@table_val.T).T
    result[np.where(result>1)] = 1
    result[np.where(result<0)] = 0

    result = invert_g(result)

    result = (result.reshape(origin_shape)*255).astype('uint8')

    return result

def lab2rgb(values: np.ndarray, illuminant: str = 'D65') -> np.ndarray:
    """
    Conversión to values Cie Lab to sRGB

    This convertion use the function lab2xyz and xyz2rgb.
    Args:
        values (np.ndarray): Values to transform
        illuminant (str, optional): standard illuminant to use['D65','D50']. Defaults to 'D65'.

    Returns:
        np.ndarray: transformed values in uint8 matrix
    """

    xyz = lab2xyz(values, illuminant)

    return xyz2rgb(xyz)
