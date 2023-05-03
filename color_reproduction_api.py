"""
Programmed by: Johan Esteban Cuervo Chica

This module 
plays color in multispectral images given in a folder
"""

import os
import re
from typing import Union
import numpy as np
import pandas as pd

from methods import transforms
from methods import color_repro as fcr
from methods.interpolate import interpolate_cie_illuminant
from methods.multispectral_image import MultiSpectralImage


class ColorReproduction:
    """
    Object contain methods for read, color reproduction, color correction
    for multispectral images
    """

    def __init__(self):
        self.image_mul = MultiSpectralImage()
        self.cie_1931 = None
        self.illuminant_d65 = None
        self.size_image = None
        self.separators = None

    def charge_cie(self) -> Union[np.ndarray, np.ndarray]:
        """
        This Function toma the wavelentghs list an charge the CIE X,Y,Z
        values.

        The wavelengths not multiply values 5. Its aproximate with linear regression.
        use the attribute self.image_mul.wavelengths for chage cie values.

        Returns:
            Union[np.ndarray, np.ndarray]: CIE_1931_table
        """

        if self.image_mul.wavelengths is None:
            raise ValueError(
                "Not inicialice variable self.image_mul.wavelengths charge"
                + " capture or asign list with values wavelengths"
            )

        name = "data/CIETABLES.xls"
        hoja = pd.read_excel(name, skiprows=4, sheet_name="Table4")
        hoja2 = pd.read_excel(name, skiprows=4, sheet_name="Table1")

        d65 = np.array(hoja2.iloc[:, [0, 2]])
        cie = np.array(hoja.iloc[:-1, :4])

        wavelengths_cie = list(cie[:, 0])
        wavelengths_d65 = list(d65[:, 0])
        self.illuminant_d65 = []
        self.cie_1931 = []

        for wavelength in self.image_mul.wavelengths:
            if wavelength > 780 or wavelength < 380:
                self.cie_1931.append([wavelength, 0, 0, 0])
                self.illuminant_d65.append([wavelength, 0])

            elif wavelength % 5 == 0:
                index = wavelengths_cie.index(wavelength)
                self.cie_1931.append(cie[index])
                index = wavelengths_d65.index(wavelength)
                self.illuminant_d65.append(d65[index])

            else:
                result = interpolate_cie_illuminant(wavelength, cie, d65)
                self.cie_1931.append([wavelength, result[0], result[1], result[2]])
                self.illuminant_d65.append([wavelength, result[3]])

        self.cie_1931 = np.array(self.cie_1931)
        self.illuminant_d65 = np.array(self.illuminant_d65)

        return self.cie_1931, self.illuminant_d65

    def read_wavelength_capture(
        self, listing: list[str], separators: Union[str, str] = None
    ) -> list[int]:
        """
        gets the wavelength of each image according to its name.
        Using a start and end separator. This separator must be special since in
        case of repetition it takes the first number.

        Args:
            listing (list[str]): names images list
            separators (Union[str, str], optional): Separator initial and final.
                It should be noted that regular expressions are used. Defaults is ['(',')'].

        Raises:
            ValueError: In case not found value read in image.

        Returns:
            list[int]: wavelengths values.
        """

        if separators is None:
            separators = [r"\(", r"\)"]

        pattern = re.compile(
            r"(?<=" + separators[0] + r")\d\d\d(?=" + separators[1] + r")"
        )
        wavelength = []
        for name in sorted(listing):
            try:
                val = int(pattern.findall(name)[0])
            except IndexError as error:
                raise ValueError(
                    "Separators not found in "
                    + f"image {name} separators: {separators}"
                ) from error

            except ValueError as error:
                raise ValueError(
                    "Separators not found in "
                    + f"image {name} separators: {separators}"
                ) from error

            wavelength.append(val)

        self.image_mul.wavelengths = wavelength

        self.charge_cie()

        return wavelength

    def convert_list_images(self, list_images: list[np.ndarray]) -> np.ndarray:
        """
        Convierte una lista de imagenes multiespectrales a una imagen de N canales
        multiespectrales.

        Args:
            list_images (list[np.ndarray]): Lista de imagenes multiespectrales

        Returns:
            np.ndarray: imagen (width, height, number channels)
        """

        array_images = list_images
        array_images = np.array(array_images)
        size = list(np.shape(array_images))
        new_size = size[1:]
        new_size.append(size[0])
        array_images = array_images.reshape((size[0], -1)).T
        array_images = array_images.reshape(new_size)

        return array_images

    def load_capture(self, path: str, up_cie: bool = True) -> MultiSpectralImage:
        """
        Load Capture MultiSpectral Image binary file comunt .micpy .

        if up_wave refresh table cie and d65

        Args:
            path (str): path file name.
            up_cie (bool, optional): update the self.image_mul.wavelengths attribute according to
            the uploaded capture. Defaults is True.
        """

        self.image_mul.load_image(path)
        self.size_image = np.shape(self.image_mul.images[0])

        if up_cie is True:
            self.charge_cie()

    def load_folder_capture(
        self, path: str, num_wave: int, start: int = 0, up_wave: bool = False
    ) -> None:
        """
        Load Capture MultiSpectral Image in folder.

        if up_wave it is necessary to preset the argument self.separators

        Args:
            path (str): Image Folder.
            num_wave (int): number of wavelengths or images per capture.
            start (int, optional): position initial of the first image. Defaults to 0.
            up_wave (bool, optional): update the self.image_mul.wavelengths attribute according to
            the uploaded capture. Defaults is True.
        """
        listing = os.listdir(path)

        listing = listing[start : start + num_wave]

        self.image_mul.images, self.size_image = fcr.read_capture(path, listing)

        if up_wave is True:
            self.image_mul.wavelengths = self.read_wavelength_capture(
                listing, self.separators
            )

    def calculate_ecualization(self, mask: np.ndarray, ideal_value: int) -> list:
        """
        calculates the equalization weights of the multispectral
        image with respect to a gray patch and an ideal value

        Args:
            mask (np.ndarray): mask patch
            ideal_value (int): ideal value gray color

        Returns:
            list: weigths values for wavelength
        """
        means = []
        for image in self.image_mul.images:
            parche = image[np.where(mask == 255)]
            mean_patch = np.mean(parche)
            means.append(mean_patch)

        equilization_w = np.divide(ideal_value * np.ones(len(means)), np.array(means))

        self.image_mul.ecualization_weigths = equilization_w
        return equilization_w

    def reproduccion_cie_1931(
        self, select_wavelengths: list[int] = None, output_color_space: str = "RGB"
    ) -> np.ndarray:
        """
        Reproduce Color CIE 1931.

        Args:
            select_wavelengths (list[int], optional): list the wavelengths per color
            reproduction. Defaults is self.image_mul.wavelengths.
            output_color_space (str): Espacio de salida ['RGB','XYZ']. to default 'RGB'
        Returns:
            image_RGB(np.ndarray): Image
        """
        matrix_images = np.zeros((0, np.size(self.image_mul.images[0])))

        for image in self.image_mul.images:
            matrix_images = np.append(matrix_images, image.reshape((1, -1)), axis=0)

        matrix_images /= 255

        if select_wavelengths is None:
            select_wavelengths = range(np.shape(matrix_images)[0])

        else:
            select_wavelengths = list(select_wavelengths)

            if len(set(select_wavelengths)) != len(select_wavelengths):
                raise ValueError(
                    f"Values repeated in select wavelentghs: {select_wavelengths}"
                )

            select_wavelengths.sort()
            index_wavelengths = []
            for wavelength in select_wavelengths:
                try:
                    index = self.image_mul.wavelengths.index(wavelength)
                except ValueError as error:
                    raise ValueError(
                        f"Not Wavelength {wavelength} "
                        + f"in self.image_mul.wavelengths : {self.image_mul.wavelengths}"
                    ) from error

                index_wavelengths.append(index)

            select_wavelengths = index_wavelengths

        if self.image_mul.ecualization_weigths is None:
            pesos_ecu = np.ones(len(select_wavelengths))

        else:
            pesos_ecu = self.image_mul.ecualization_weigths[select_wavelengths]

        # Coeficientes
        esc = (np.ones((3, 1)) * self.illuminant_d65[select_wavelengths, 1].T).T
        coef = (self.cie_1931[select_wavelengths, 1:] * esc).T
        sum_n = np.sum(coef, axis=1)

        shape_imag = list(self.size_image)
        shape_imag.append(3)
        # Reproduccion de color usando CIE

        xyz = np.dot(coef, (matrix_images[select_wavelengths, :].T * pesos_ecu).T).T
        xyz = xyz / sum_n[1]

        if output_color_space.upper() == "XYZ":
            return np.reshape(xyz, shape_imag)

        elif output_color_space.upper() == "RGB":
            rgb = transforms.xyz2rgb(xyz)
            return np.reshape(rgb, shape_imag).astype("uint8")
        else:
            raise ValueError(
                f"No color space output: {output_color_space}. or 'RGB','XYZ'"
            )
