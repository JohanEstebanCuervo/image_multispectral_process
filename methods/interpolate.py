import numpy as np


def interpolate_cie_illuminant(
    wavelength: int, cie: np.ndarray, illuminant: np.ndarray
) -> list:
    """
    Calculate te values for X,Y,Z and illuminant d65.
    for wavelength.

    The values the matrix wavelengths deben is multiply the 5
    Args:
        wavelength (int): value the waneltngth entry 380, 780
        cie (np.ndarray): Matrix N*4 with wavelength, X, Y, Z
        illuminant (np.ndarray): Matrix N*2 with wavelength, Illuminant

    Returns:
        list: [X, Y, Z, Illuminant]
    """

    result = []
    min_wav = (wavelength // 5) * 5
    max_wav = min_wav + 5

    wavelengths_cie = list(cie[:, 0])
    values_cie_min = cie[wavelengths_cie.index(min_wav), :]
    values_cie_max = cie[wavelengths_cie.index(max_wav), :]

    for i in range(1, 4):
        pendient = (values_cie_max[i] - values_cie_min[i]) / (max_wav - min_wav)
        intercept = -pendient * min_wav + values_cie_min[i]

        interpolate = pendient * wavelength + intercept
        result.append(interpolate)

    wavelengths_illuminant = list(illuminant[:, 0])
    values_illuminant_min = illuminant[wavelengths_illuminant.index(min_wav), :]
    values_illuminant_max = illuminant[wavelengths_illuminant.index(max_wav), :]

    pendient = (values_illuminant_max[1] - values_illuminant_min[1]) / (
        max_wav - min_wav
    )
    intercept = -pendient * min_wav + values_illuminant_min[1]

    interpolate = pendient * wavelength + intercept
    result.append(interpolate)

    return result


def linear_regression(points_x: list, points_y: list) -> tuple[float, float]:
    """
    function that calculates a linear regression through least squares

        Args:
            points_x (list): points x
            points_y (list): points y

        Returns:
            tuple[float, float]: slope, intercept
    """

    mat_x = np.ones((len(points_x), 2))
    mat_x[:, 0] = points_x
    mat_y = np.array(points_y).reshape((-1, 1))
    inv_x = np.linalg.pinv(mat_x)

    result = inv_x @ mat_y

    return result.squeeze()
