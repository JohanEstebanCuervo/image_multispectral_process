"""
Programmed by: Johan Esteban Cuervo Chica

This module includes color space transformations and
matricial transformations images
"""

import numpy as np
import cv2


def function_f(arg_x: np.ndarray | float) -> np.ndarray | float:
    """
    Function invert f for convert xyz to lab
    supports float values or list of values

    Args:
        arg_x (list): values

    Returns:
        list: values
    """
    delta = (6 / 29) ** 3

    try:
        arg_x = float(arg_x)

        if arg_x > delta:
            return arg_x ** (1 / 3)

        else:

            return arg_x / (3 * (6 / 29) ** 2) + 4 / 29
    except TypeError:
        pass

    result = np.zeros_like(arg_x, dtype="float")

    max_index = np.where(arg_x > delta)
    min_index = np.where(arg_x <= delta)

    result[max_index] = np.power(arg_x[max_index], 1 / 3)
    result[min_index] = np.divide(arg_x[min_index], (3 * (6 / 29) ** 2)) + 4 / 29

    return result


def invert_f(arg_x: np.ndarray | float) -> np.ndarray | float:
    """
    Function invert f for convert lab to xyz
    supports float values or list of values

    Args:
        arg_x (list): values

    Returns:
        list: values
    """
    delta = 6 / 29

    try:
        arg_x = float(arg_x)

        if arg_x > delta:
            return arg_x**3

        else:

            return 3 * (delta**2) * (arg_x - 4 / 29)

    except TypeError:
        pass

    result = np.zeros_like(arg_x, dtype="float")

    max_index = np.where(arg_x > delta)
    min_index = np.where(arg_x <= delta)

    result[max_index] = np.power(arg_x[max_index], 3)
    result[min_index] = 3 * (delta**2) * (arg_x[min_index] - 4 / 29)

    return result


def function_g(arg_x: np.ndarray | int) -> np.ndarray | float:
    """
    Function invert f for convert rgb to xyz
    supports float values or list of values

    Args:
        arg_x (list): values [0 - 255]

    Returns:
        list: values
    """
    delta = 0.04045
    alpha = 0.055
    gamma = 2.4

    try:
        arg_x = float(arg_x) / 255
        if arg_x > 1:
            arg_x = 1

        if arg_x > delta:
            return np.power((arg_x + alpha) / 1.055, gamma)

        else:

            return arg_x / 12.92

    except TypeError:
        pass

    result = np.zeros_like(arg_x, dtype="float")

    arg_x = arg_x.astype("float") / 255
    max_index = np.where(arg_x > delta)
    min_index = np.where(arg_x <= delta)
    result[max_index] = np.power(np.divide((arg_x[max_index] + alpha), 1.055), gamma)
    result[min_index] = arg_x[min_index] / 12.92

    return result


def invert_g(arg_x: np.ndarray | float) -> np.ndarray | float:
    """
    Function invert f for convert xyz to rgb
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
            return (1 + alpha) * np.power(arg_x, 1 / gamma) - alpha

        else:

            return 12.92 * arg_x

    except TypeError:
        pass

    result = np.zeros_like(arg_x, dtype="float")

    arg_x[np.where(arg_x > 1)] = 1
    max_index = np.where(arg_x > delta)
    min_index = np.where(arg_x <= delta)

    result[max_index] = (1 + alpha) * np.power(arg_x[max_index], 1 / gamma) - alpha
    result[min_index] = 12.92 * arg_x[min_index]

    return result


def xyz2lab(values: np.ndarray, illuminant: str = "D65") -> np.ndarray:
    """
    Conversión to values XYZ to Cie Lab

    Args:
        values (np.ndarray): Values to transform values entry 0,1
        illuminant (str, optional): standard illuminant to use['D65','D50']. Defaults to 'D65'.

    Returns:
        np.ndarray: transformed values
    """

    if illuminant.upper() == "D65":
        xyzn = [95.0489, 100, 108.8840]

    elif illuminant.upper() == "D50":
        xyzn = [96.4212, 100, 82.5188]

    else:
        raise ValueError(
            f"Illuminant incorrect options: 'D65','D50' joined: {illuminant}"
        )

    values = np.array(values)
    origin_shape = np.shape(values)

    table_val = values.reshape((-1, 3)) / xyzn * 100

    table_val = function_f(table_val)

    result = np.zeros_like(table_val, dtype="float")

    result[:, 0] = 116 * table_val[:, 1] - 16
    result[:, 1] = 500 * (table_val[:, 0] - table_val[:, 1])
    result[:, 2] = 200 * (table_val[:, 1] - table_val[:, 2])

    return result.reshape(origin_shape)


def lab2xyz(values: np.ndarray, illuminant: str = "D65") -> np.ndarray:
    """
    Conversión to values Cie Lab to XYZ

    Args:
        values (np.ndarray): Values to transform
        illuminant (str, optional): standard illuminant to use['D65','D50']. Defaults to 'D65'.

    Returns:
        np.ndarray: transformed values
    """

    if illuminant.upper() == "D65":
        xyzn = [95.0489, 100, 108.8840]

    elif illuminant.upper() == "D50":
        xyzn = [96.4212, 100, 82.5188]

    else:
        raise ValueError(
            f"Illuminant incorrect options: 'D65','D50' joined: {illuminant}"
        )

    values = np.array(values)
    origin_shape = np.shape(values)

    table_val = values.reshape((-1, 3))

    result = np.zeros_like(table_val, dtype="float")

    result[:, 0] = (table_val[:, 0] + 16) / 116 + table_val[:, 1] / 500
    result[:, 1] = (table_val[:, 0] + 16) / 116
    result[:, 2] = (table_val[:, 0] + 16) / 116 - table_val[:, 2] / 200

    result = invert_f(result)

    result *= xyzn

    return result.reshape(origin_shape) / 100


def xyz2rgb(values: np.ndarray) -> np.ndarray:
    """
    Conversión to values Cie XYZ to sRGB Values

    Args:
        values (np.ndarray): Values to transform

    Returns:
        np.ndarray: transformed values. Matrix uint8
    """

    matrix = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )

    values = np.array(values)
    origin_shape = np.shape(values)

    table_val = values.reshape((-1, 3))

    result = (matrix @ table_val.T).T
    result[np.where(result > 1)] = 1
    result[np.where(result < 0)] = 0

    result = invert_g(result)

    result = (result.reshape(origin_shape) * 255).astype("uint8")

    return result


def rgb2xyz(values: np.ndarray) -> np.ndarray:
    """
    Conversión to values sRGB to Cie XYZ Values

    Args:
        values (np.ndarray): Values to transform

    Returns:
        np.ndarray: transformed values. Matrix float
    """

    matrix = np.array(
        [
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ]
    )

    values = np.array(values)
    origin_shape = np.shape(values)

    table_val = values.reshape((-1, 3))

    table_val = function_g(table_val)

    result = (matrix @ table_val.T).T

    result = result.reshape(origin_shape)

    return result


def lab2rgb(values: np.ndarray, illuminant: str = "D65") -> np.ndarray:
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


def rgb2lab(values: np.ndarray, illuminant: str = "D65") -> np.ndarray:
    """
    Conversión to values sRGB to CieLab

    This convertion use the function rgb2xyz and xyz2lab.
    Args:
        values (np.ndarray): Values to transform
        illuminant (str, optional): standard illuminant to use['D65','D50']. Defaults to 'D65'.

    Returns:
        np.ndarray: transformed values in uint8 matrix
    """

    xyz = rgb2xyz(values)

    return xyz2lab(xyz, illuminant=illuminant)


def rotate_image(
    image: np.ndarray,
    angle: float,
    axis: tuple[int, int],
    size_out: tuple[int, int] = None,
) -> np.ndarray:
    """
    Rotate image

    Args:
        image (np.ndarray): image array
        angle (float): angle to rotate (rad)
        axis (tuple[int, int]): axis of rotation
        size_out (tuple[int, int], optional): Size image out (Style Numpy).
            Defaults image input shape.

    Returns:
        np.ndarray: _description_
    """
    if size_out is None:
        size_out = np.flip(np.shape(image)[:2])

    alpha = np.cos(angle).reshape(1)
    beta = np.sin(angle).reshape(1)

    matriz_rotate = np.array(
        [
            [alpha, beta, (1 - alpha) * axis[1] - beta * axis[0]],
            [-beta, alpha, beta * axis[1] + (1 - alpha) * axis[0]],
        ]
    ).reshape((2, 3))

    return cv2.warpAffine(image, matriz_rotate, size_out)
