"""
Programmed by: Johan Esteban Cuervo Chica

This module 
plays color in multispectral images given in a folder
"""

import os
from typing import Union
import numpy as np
import pandas as pd

from methods import transforms
from methods import color_repro as fcr
from methods.interpolate import interpolate_cie_illuminant

class ColorReproduction():

    """docstring for ColorReprocution"""

    def __init__(self):

        self.images = None
        self.mask = None
        self.wavelengths = None
        self.cie_1931 = None
        self.illuminant_d65 = None
        self.size_image = None
        self.pesos_ecu = None
        self.separators = None
        self.transforms = transforms

    def charge_cie(self) -> Union[np.ndarray, np.ndarray]:
        """
        This Function toma the wavelentghs list an charge the CIE X,Y,Z
        values.

        The wavelengths not multiply values 5. Its aproximate with linear regression

        Returns:
            Union[np.ndarray, np.ndarray]: _description_
        """
        if self.wavelengths is None:
            raise ValueError('Not inicialice variable self.wavelengths charge'
                             + ' capture or asign list with values wavelengths')

        name = 'data/CIETABLES.xls'
        hoja = pd.read_excel(name, skiprows=4, sheet_name='Table4')
        hoja2 = pd.read_excel(name, skiprows=4, sheet_name='Table1')

        d65 = np.array(hoja2.iloc[:, [0, 2]])
        cie = np.array(hoja.iloc[: - 1, :4])

        wavelengths_cie = list(cie[:, 0])
        wavelengths_d65 = list(d65[:, 0])
        self.illuminant_d65 = []
        self.cie_1931 = []

        for wavelength in self.wavelengths:

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

    def load_capture(self, path: str, num_wave: int, start: int =0) -> None:
        """
        Load Capture MultiSpectral Image in folder.

        Args:
            path (str): Image Folder 
            num_wave (int): number of wavelengths or images per capture
            start (int, optional): position initial of the first image. Defaults to 0.
        """
        listing = os.listdir(path)

        listing = listing[start:start + num_wave]

        self.images, self.size_image = fcr.read_capture(path, listing)

        if self.wavelengths is None:
            self.wavelengths = fcr.read_wavelength_capture(listing, self.separators)

        self.charge_cie()

    def reproduccion_cie_1931(self, select_wavelengths: list[int] = None) -> np.ndarray:
        """
        Reproduce Color CIE 1931. 

        Args:
            select_wavelengths (list[int], optional): list the wavelengths per color
            reproduction. Defaults is self.wavelengths.

        Returns:
            image_RGB(np.ndarray): Image in RGB type uint8
        """

        if select_wavelengths is None:

            select_wavelengths = range(np.shape(self.images)[0])

        else:
            select_wavelengths = list(select_wavelengths)

            if len(set(select_wavelengths)) != len(select_wavelengths):
                raise ValueError(f'Values repeated in select wavelentghs: {select_wavelengths}')

            select_wavelengths.sort()
            index_wavelengths = []
            for wavelength in select_wavelengths:
                try:
                    index = self.wavelengths.index(wavelength)
                except ValueError as error:
                    raise ValueError(f'Not Wavelength {wavelength} ' +
                                    f'in self.wavelengths : {self.wavelengths}') from error

                index_wavelengths.append(index)

            select_wavelengths = index_wavelengths

        if self.pesos_ecu is None:
            pesos_ecu = np.ones(len(select_wavelengths))

        else:
            pesos_ecu = self.pesos_ecu[select_wavelengths]

        # Coeficientes
        esc = (np.ones((3, 1)) * self.illuminant_d65[select_wavelengths, 1].T).T
        coef = (self.cie_1931[select_wavelengths, 1:] * esc).T
        sum_n = np.sum(coef, axis=1)

        # Reproduccion de color usando CIE

        xyz = np.dot(coef, (self.images[select_wavelengths, :].T * pesos_ecu).T).T
        xyz = xyz / sum_n[1]

        rgb = transforms.xyz2rgb(xyz)
        shape_imag = list(self.size_image)
        shape_imag.append(3)
        im_rgb = np.reshape(rgb, shape_imag)

        return im_rgb
