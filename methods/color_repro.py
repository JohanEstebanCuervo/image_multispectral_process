"""
Programmed by: Johan Esteban Cuervo Chica

"""
import pickle
import itertools
import re
from typing import Union

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# Lectura de masks y colocación en una listing(cant masks)
def ext_masks(folder: str, listing: list[str]) -> list[np.ndarray]:
    """
    read the color checker patch masks

    Args:
        folder (str): folder path
        listing (list[str]): list maks

    Returns:

        list[np.ndarray]: list the masks
    """
    masks = []
    for name in sorted(listing):
        mask = cv2.imread(folder + "/" + name, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        masks.append(np.array(mask))
    return masks


def read_capture(folder: str, listing: list[str]) -> Union[np.ndarray, tuple]:
    """
    read a list of images from a folder,
    groups them into an array of number_pixels * number_images

    Args:
        folder (str): folder path
        listing (list[str]): names images list

    Returns:
        Union[np.ndarray, tuple]: matrix the images numpixels* number_images, shape origin image
    """
    imagenespatron = []
    for name in sorted(listing):
        imagen = cv2.imread(folder + "/" + name, cv2.IMREAD_GRAYSCALE)

        # se convierte la imagen en una columna y se concatena con las demas del espectro
        imagenespatron = np.concatenate(
            (imagenespatron, np.reshape(imagen, (-1))), axis=0
        )

    shape_imag = np.shape(imagen)

    # se redimensiona a  Filas * N imagenes multiespectrales
    imagenespatron = imagenespatron.reshape(len(listing), -1)

    return imagenespatron / 255, shape_imag


def read_wavelength_capture(
    listing: list[str], separators: Union[str, str] = None
) -> list[int]:
    """
    gets the wavelength of each image according to its name.
    Using a start and end separator. This separator must be special since in
    case of repetition it takes the first number.

    Args:
        listing (list[str]): names images list
        separators (Union[str, str], optional): Separator initial and final.
            It should be noted that regular expressions are used. Defaults is ['\(','\)'].

    Raises:
        ValueError: In case not found value read in image.

    Returns:
        list[int]: wavelengths values.
    """

    if separators is None:
        separators = [r"\(", r"\)"]

    pattern = re.compile(r"(?<=" + separators[0] + r")\d\d\d(?=" + separators[1] + r")")
    wavelength = []
    for name in sorted(listing):
        try:
            val = int(pattern.findall(name)[0])
        except IndexError as error:
            raise ValueError(
                "Separators not found in " + f"image {name} separators: {separators}"
            ) from error

        except ValueError as error:
            raise ValueError(
                "Separators not found in " + f"image {name} separators: {separators}"
            ) from error

        wavelength.append(val)

    return wavelength


def ideal_color_patch_pixel(color_check, masks):

    color_ipmask = [[0, 0, 0]]
    for i, mask in enumerate(masks):  # se recorren las masks
        N = np.shape(np.where(mask == 255))[1]
        color = np.concatenate(
            (
                color_check[i][0] * np.ones(N),
                color_check[i][1] * np.ones(N),
                color_check[i][2] * np.ones(N),
            ),
            axis=0,
        )  # un vector columna con los valores RGB ideales de cada parche N pixeles de parche
        color = color.reshape(3, -1).T  # redimensiona
        color_ipmask = np.concatenate(
            (color_ipmask, color), axis=0
        )  # concatena el color ideal de los 24 parches
    color_ipmask = color_ipmask[1:, :]  # se borra la primer fila que son 0
    return color_ipmask


def equalization_weights(
    pattern_capture: np.ndarray, mask: np.ndarray, ideal_value: int = 243
) -> list[float]:
    """
    Calculate equalization weights for capture multispectral
    according to a reference gray patch given on the mask

    Args:
        pattern_capture (np.ndarray): capture multiespectral matrix
        mask (np.ndarray): reference pixels
        ideal_value (int, optional): value ideal for reference pixels. Defaults to 243.

    Returns:
        list[float]: equalization weights
    """

    promedios = []
    mask = np.reshape(mask, (-1))
    for image in pattern_capture:
        parche = image[np.where(mask == 255)]
        prom = np.mean(parche)
        promedios.append(prom)

    equilization_w = np.divide(
        ideal_value * np.ones(len(promedios)), np.array(promedios)
    )
    return equilization_w


def promedio_RGB_parches(Irecons_RGB, masks):
    prom = []
    for i in range(len(masks)):
        R, G, B = (
            Irecons_RGB[:, :, 0],
            Irecons_RGB[:, :, 1],
            Irecons_RGB[:, :, 2],
        )  # Se separa la imagen EN R,G,B
        parte = R[np.where(masks[i] == 255)]  # Parte R del parche
        prom.append(np.mean(parte))  # Media parte R concatenada a promedios
        parte = G[np.where(masks[i] == 255)]
        prom.append(np.mean(parte))
        parte = B[np.where(masks[i] == 255)]
        prom.append(np.mean(parte))

    return np.reshape(prom, (24, 3))  # se redimenciona los promedio a un array 24,3


# Extrae los valores RGB de los parches para realizar alguna regresión
def RGB_IN(Irecons_RGB, masks):
    parches_r = []
    parches_g = []
    parches_b = []

    for i in range(len(masks)):
        R, G, B = Irecons_RGB[:, :, 0], Irecons_RGB[:, :, 1], Irecons_RGB[:, :, 2]
        parte = R[np.where(masks[i] == 255)]
        parches_r = np.concatenate((parches_r, parte))
        parte = G[np.where(masks[i] == 255)]
        parches_g = np.concatenate((parches_g, parte))
        parte = B[np.where(masks[i] == 255)]
        parches_b = np.concatenate((parches_b, parte))

    parches_rgb = np.zeros((len(parches_r), 3))
    parches_rgb[:, 0] = parches_r
    parches_rgb[:, 1] = parches_g
    parches_rgb[:, 2] = parches_b
    return parches_rgb


def RGB_IN_mean(Irecons_RGB, masks):
    parches_rgb = []

    for i in range(len(masks)):
        R, G, B = Irecons_RGB[:, :, 0], Irecons_RGB[:, :, 1], Irecons_RGB[:, :, 2]
        parte = R[np.where(masks[i] == 255)]
        parches_rgb.append(np.mean(parte))
        parte = G[np.where(masks[i] == 255)]
        parches_rgb.append(np.mean(parte))
        parte = B[np.where(masks[i] == 255)]
        parches_rgb.append(np.mean(parte))

    return np.reshape(parches_rgb, (-1, 3))


# parches pixeles Imagen de infrarrojo cercano tomada con la imagen numero 14 lambda 840 nm
def N_IN(Irecons, masks):
    parches = []
    for i in range(len(masks)):
        parte = Irecons[np.where(masks[i] == 255)]
        parches = np.concatenate((parches, parte))

    return np.reshape(parches, (1, -1))


# condicionales de valores limites de imagenes despues de una transformación
def recorte(im):
    im[np.where(im > 1)] = 1
    im[np.where(im < 0)] = 0

    return im


# offset de imagenes para transformaciones logaritmicas fijando valor minimo a 1/255
def offset(im):
    im[np.where(im <= 1 / 255)] = 1 / 255

    return im


# funcion para mostrar imagenes con matplotlib  con rango de flotantes (0 a  1)
def imshow(titulo, imagen):

    if len(np.shape(imagen)) == 2:
        imagen1 = np.zeros((np.shape(imagen)[0], np.shape(imagen)[1], 3))
        imagen1[:, :, 0] = imagen
        imagen1[:, :, 1] = imagen
        imagen1[:, :, 2] = imagen
        imagen = imagen1

    plt.imshow(imagen)
    plt.title(titulo)
    plt.axis("off")
    plt.show()


# error de reproduccion distancia euclidea  pixel por pixel de cada parche
# y promedio de error por parche para multiples imagenes reconstruidas


def Error_de_reproduccion(imagenes_RGB, masks, color_check):
    error = []
    for imagen in imagenes_RGB:
        imagen = (imagen.reshape(-1, 3) * 255).astype(int)
        for i, mask in enumerate(masks):
            indices = np.where(
                mask.reshape(
                    -1,
                )
                == 255
            )
            dif = imagen[indices] - color_check[i]
            DistEucl = np.sqrt(np.sum(np.power(dif, 2), axis=1))
            error.append(np.mean(DistEucl))
    return np.reshape(error, (-1, len(masks)))


#%% Funciones CCM para una imagen
# Color Correction matriz linear
def CCM_Linear(im_RGB, colorn, masks, shape_imag=(480, 640, 3)):
    entrada = RGB_IN(im_RGB, masks).T
    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = colorn.T / 255

    PseudoINV = np.linalg.pinv(entrada)

    Ccm = np.dot(colorn, PseudoINV)
    rgb = np.reshape(im_RGB, (-1, 3)).T
    rgb = np.concatenate((rgb, np.ones((1, np.shape(rgb)[1]))))
    rgb = np.dot(Ccm, rgb)
    im_rgb = np.reshape(rgb.T, shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb, Ccm


# Color Correction matriz Compound
def CCM_Compound(im_RGB, colorn, masks, shape_imag=(480, 640, 3)):
    entrada = RGB_IN(im_RGB, masks).T
    # entradaN = N_IN(N, masks)
    # entrada = np.concatenate((entrada,entradaN))
    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = np.log(colorn.T / 255)

    PseudoINV = np.linalg.pinv(entrada)

    Ccm = np.dot(colorn, PseudoINV)
    rgb = np.reshape(im_RGB, (-1, 3)).T
    # rgbn= np.concatenate((rgb,np.reshape(N,(1,-1))))
    rgb = np.concatenate((rgb, np.ones((1, np.shape(rgb)[1]))))

    lnrgb = np.dot(Ccm, rgb)
    rgb = np.exp(lnrgb)
    im_rgb = np.reshape(rgb.T, shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb, Ccm


# Color Correction matriz Logarithm
def CCM_Logarithm(im_RGB, colorn, masks, shape_imag=(480, 640, 3)):
    entrada = np.log(offset(RGB_IN(im_RGB, masks).T))
    # entradaN = np.log(offset(N_IN(N, masks)))
    # entrada = np.concatenate((entrada,entradaN))
    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = colorn.T / 255

    PseudoINV = np.linalg.pinv(entrada)

    Ccm = np.dot(colorn, PseudoINV)
    lnrgb = np.log(offset(np.reshape(im_RGB, (-1, 3)).T))
    # lnrgbn = np.concatenate((lnrgb,np.log(offset(np.reshape(N,(1,-1))))))
    lnrgb = np.concatenate((lnrgb, np.ones((1, np.shape(lnrgb)[1]))))

    rgb = np.dot(Ccm, lnrgb)
    im_rgb = np.reshape(rgb.T, shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb, Ccm


# Color Correction matriz Polynomial Con NIR
def CCM_Polynomial_N(im_RGB, N, colorn, masks, shape_imag=(480, 640, 3)):

    entrada = RGB_IN(im_RGB, masks).T
    entradaN = N_IN(N, masks)
    entrada = np.concatenate((entrada, entradaN))

    R2 = entrada[0, :] ** 2
    G2 = entrada[1, :] ** 2
    B2 = entrada[2, :] ** 2
    N2 = entradaN**2
    RG = entrada[0, :] * entrada[1, :]
    RB = entrada[0, :] * entrada[2, :]
    RN = entrada[0, :] * entradaN
    GB = entrada[1, :] * entrada[2, :]
    GN = entrada[1, :] * entradaN
    BN = entrada[2, :] * entradaN

    entrada = np.concatenate((entrada, [R2]))
    entrada = np.concatenate((entrada, [G2]))
    entrada = np.concatenate((entrada, [B2]))
    entrada = np.concatenate((entrada, N2))
    entrada = np.concatenate((entrada, [RG]))
    entrada = np.concatenate((entrada, [RB]))
    entrada = np.concatenate((entrada, RN))
    entrada = np.concatenate((entrada, [GB]))
    entrada = np.concatenate((entrada, GN))
    entrada = np.concatenate((entrada, BN))
    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = colorn.T / 255

    PseudoINV = np.linalg.pinv(entrada)

    entradaN = np.reshape(N, (1, -1))

    Ccm = np.dot(colorn, PseudoINV)
    rgb = np.reshape(im_RGB, (-1, 3)).T
    rgb = np.concatenate((rgb, entradaN))
    R2 = rgb[0, :] ** 2
    G2 = rgb[1, :] ** 2
    B2 = rgb[2, :] ** 2
    N2 = entradaN**2
    RG = rgb[0, :] * rgb[1, :]
    RB = rgb[0, :] * rgb[2, :]
    RN = rgb[0, :] * entradaN
    GB = rgb[1, :] * rgb[2, :]
    GN = rgb[1, :] * entradaN
    BN = rgb[2, :] * entradaN

    rgb = np.concatenate((rgb, [R2]))
    rgb = np.concatenate((rgb, [G2]))
    rgb = np.concatenate((rgb, [B2]))
    rgb = np.concatenate((rgb, N2))
    rgb = np.concatenate((rgb, [RG]))
    rgb = np.concatenate((rgb, [RB]))
    rgb = np.concatenate((rgb, RN))
    rgb = np.concatenate((rgb, [GB]))
    rgb = np.concatenate((rgb, GN))
    rgb = np.concatenate((rgb, BN))
    rgb = np.concatenate((rgb, np.ones((1, np.shape(rgb)[1]))))

    rgb = np.dot(Ccm, rgb)
    im_rgb = np.reshape(rgb.T, (480, 640, 3))
    im_rgb = recorte(im_rgb)
    return im_rgb, Ccm, R2


# Color Correction matriz Polynomial
def CCM_Polynomial(im_RGB, colorn, masks, shape_imag=(480, 640, 3)):

    entrada = RGB_IN(im_RGB, masks).T

    R2 = entrada[0, :] ** 2
    G2 = entrada[1, :] ** 2
    B2 = entrada[2, :] ** 2
    RG = entrada[0, :] * entrada[1, :]
    RB = entrada[0, :] * entrada[2, :]
    GB = entrada[1, :] * entrada[2, :]

    entrada = np.concatenate((entrada, [R2]))
    entrada = np.concatenate((entrada, [G2]))
    entrada = np.concatenate((entrada, [B2]))
    entrada = np.concatenate((entrada, [RG]))
    entrada = np.concatenate((entrada, [RB]))
    entrada = np.concatenate((entrada, [GB]))

    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = colorn.T / 255

    PseudoINV = np.linalg.pinv(entrada)

    Ccm = np.dot(colorn, PseudoINV)
    rgb = np.reshape(im_RGB, (-1, 3)).T
    R2 = rgb[0, :] ** 2
    G2 = rgb[1, :] ** 2
    B2 = rgb[2, :] ** 2
    RG = rgb[0, :] * rgb[1, :]
    RB = rgb[0, :] * rgb[2, :]
    GB = rgb[1, :] * rgb[2, :]

    rgb = np.concatenate((rgb, [R2]))
    rgb = np.concatenate((rgb, [G2]))
    rgb = np.concatenate((rgb, [B2]))
    rgb = np.concatenate((rgb, [RG]))
    rgb = np.concatenate((rgb, [RB]))
    rgb = np.concatenate((rgb, [GB]))
    rgb = np.concatenate((rgb, np.ones((1, np.shape(rgb)[1]))))

    rgb = np.dot(Ccm, rgb)
    im_rgb = np.reshape(rgb.T, shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb, Ccm, R2


#%% FUNCIONES CCM Para multiples imagenes
# Color Correction matriz linear
def CCM_Linear_Train(archivo):
    datatrain = pd.read_csv(archivo, sep=",", names=range(1, 7))
    datatrain = datatrain.to_numpy()
    entrada = datatrain[:, :3].T / 255
    colorn = datatrain[:, 3:]
    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = colorn.T / 255

    PseudoINV = np.linalg.pinv(entrada)
    Ccm_Linear = np.dot(colorn, PseudoINV)

    return Ccm_Linear


# Color Correction matriz Compound
def CCM_Compound_Train(archivo):
    datatrain = pd.read_csv(archivo, sep=",", names=range(1, 7))
    datatrain = datatrain.to_numpy()
    entrada = datatrain[:, :3].T / 255
    colorn = datatrain[:, 3:]
    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = np.log(colorn.T / 255)

    PseudoINV = np.linalg.pinv(entrada)

    Ccm_Compound = np.dot(colorn, PseudoINV)

    return Ccm_Compound


# Color Correction matriz Logarithm
def CCM_Logarithm_Train(archivo):
    datatrain = pd.read_csv(archivo, sep=",", names=range(1, 7))
    datatrain = datatrain.to_numpy()
    entrada = datatrain[:, :3].T / 255
    colorn = datatrain[:, 3:]
    entrada = np.log(offset(entrada))
    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = colorn.T / 255

    PseudoINV = np.linalg.pinv(entrada)

    Ccm_Logatirhm = np.dot(colorn, PseudoINV)

    return Ccm_Logatirhm


# Color Correction matriz Polynomial
def CCM_Polynomial_Train(archivo):
    datatrain = pd.read_csv(archivo, sep=",", names=range(1, 7))
    datatrain = datatrain.to_numpy()
    entrada = datatrain[:, :3].T / 255
    colorn = datatrain[:, 3:]

    R2 = entrada[0, :] ** 2
    G2 = entrada[1, :] ** 2
    B2 = entrada[2, :] ** 2
    RG = entrada[0, :] * entrada[1, :]
    RB = entrada[0, :] * entrada[2, :]
    GB = entrada[1, :] * entrada[2, :]

    entrada = np.concatenate((entrada, [R2]))
    entrada = np.concatenate((entrada, [G2]))
    entrada = np.concatenate((entrada, [B2]))
    entrada = np.concatenate((entrada, [RG]))
    entrada = np.concatenate((entrada, [RB]))
    entrada = np.concatenate((entrada, [GB]))

    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = colorn.T / 255

    PseudoINV = np.linalg.pinv(entrada)

    Ccm_Polynomial = np.dot(colorn, PseudoINV)

    return Ccm_Polynomial


#%% CCM test
# Color Correction matriz linear
def CCM_Linear_Test(im_RGB, Ccm):
    shape_imag = np.shape(im_RGB)
    rgb = np.reshape(im_RGB, (-1, 3)).T
    rgb = np.concatenate((rgb, np.ones((1, np.shape(rgb)[1]))))
    rgb = np.dot(Ccm, rgb)
    im_rgb = np.reshape(rgb.T, shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb


# Color Correction matriz Compound
def CCM_Compound_Test(im_RGB, Ccm):
    shape_imag = np.shape(im_RGB)
    rgb = np.reshape(im_RGB, (-1, 3)).T
    rgb = np.concatenate((rgb, np.ones((1, np.shape(rgb)[1]))))

    lnrgb = np.dot(Ccm, rgb)
    rgb = np.exp(lnrgb)
    im_rgb = np.reshape(rgb.T, shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb


# Color Correction matriz Logarithm
def CCM_Logarithm_Test(im_RGB, Ccm):
    shape_imag = np.shape(im_RGB)
    lnrgb = np.log(offset(np.reshape(im_RGB, (-1, 3)).T))

    lnrgb = np.concatenate((lnrgb, np.ones((1, np.shape(lnrgb)[1]))))

    rgb = np.dot(Ccm, lnrgb)
    im_rgb = np.reshape(rgb.T, shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb


# Color Correction matriz Polynomial
def CCM_Polynomial_Test(im_RGB, Ccm):
    shape_imag = np.shape(im_RGB)
    rgb = np.reshape(im_RGB, (-1, 3)).T
    R2 = rgb[0, :] ** 2
    G2 = rgb[1, :] ** 2
    B2 = rgb[2, :] ** 2
    RG = rgb[0, :] * rgb[1, :]
    RB = rgb[0, :] * rgb[2, :]
    GB = rgb[1, :] * rgb[2, :]

    rgb = np.concatenate((rgb, [R2]))
    rgb = np.concatenate((rgb, [G2]))
    rgb = np.concatenate((rgb, [B2]))
    rgb = np.concatenate((rgb, [RG]))
    rgb = np.concatenate((rgb, [RB]))
    rgb = np.concatenate((rgb, [GB]))
    rgb = np.concatenate((rgb, np.ones((1, np.shape(rgb)[1]))))

    rgb = np.dot(Ccm, rgb)
    im_rgb = np.reshape(rgb.T, shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb


#%%
# Generacion de imagen con parches ideales y reproducidos.


def imwrite(titulo, imagen):
    imagen = np.array(imagen * 255, dtype="uint8")
    imagen2 = np.copy(imagen)
    imagen[:, :, 0] = imagen2[:, :, 2]
    imagen[:, :, 2] = imagen2[:, :, 0]
    cv2.imwrite(titulo, imagen)


def comparacion_color_check(name, im_RGB, color_check_RGB, masks, folder=""):

    Grosor = 2

    for i in range(4):
        fila = np.zeros((60, Grosor, 3))
        for j in range(6):
            parchei = np.ones((60, 60, 3)) * color_check_RGB[6 * i + j, :]
            fila = np.concatenate((fila, parchei), axis=1)
            fila = np.concatenate((fila, np.zeros((60, Grosor, 3))), axis=1)

        if i == 0:
            imagen = np.zeros((Grosor, np.shape(fila)[1], 3))

        imagen = np.concatenate((imagen, fila), axis=0)

        fila = np.zeros((60, Grosor, 3))
        for j in range(6):

            parchei = im_RGB[np.where(255 == masks[6 * i + j])] * 255
            if len(np.where(255 == masks[6 * i + j])[0]) < 3600:
                longitud = 3600 - len(np.where(255 == masks[6 * i + j])[0])
                parchei = np.concatenate((parchei, parchei[:longitud, :]))
            if len(np.where(255 == masks[6 * i + j])[0]) > 3600:
                parchei = parchei[:3600, :]
            parchei = np.reshape(parchei, (60, 60, 3)).astype(int)
            fila = np.concatenate((fila, parchei), axis=1)
            fila = np.concatenate((fila, np.zeros((60, Grosor, 3))), axis=1)

        imagen = np.concatenate((imagen, fila), axis=0)
        imagen = np.concatenate(
            (imagen, np.zeros((Grosor, np.shape(fila)[1], 3))), axis=0
        )

    imshow(folder + "/Comparación Color_Check - " + name, imagen.astype(int))
    imwrite(folder + "/Comparacion Color_Check - " + name + ".png", imagen / 255)


#%%


def ReproduccionCie19312(
    imagenes_patron, Pesos_ecu, shape_imag=(480, 640, 3), select_wavelengths="All"
):

    if select_wavelengths == "All":
        select_wavelengths = range(np.shape(imagenes_patron)[0])

    D65 = np.array(
        [
            [410, 91.486000],
            [450, 117.008000],
            [470, 114.861000],
            [490, 108.811000],
            [505, 108.578000],
            [530, 107.689000],
            [560, 100.000000],
            [590, 88.685600],
            [600, 90.006200],
            [620, 87.698700],
            [630, 83.288600],
            [650, 80.026800],
            [720, 61.604000],
        ]
    )

    CIE1931 = np.array(
        [
            [410, 0.043510, 0.001210, 0.207400],
            [450, 0.336200, 0.038000, 1.772110],
            [470, 0.195360, 0.090980, 1.287640],
            [490, 0.032010, 0.208020, 0.465180],
            [505, 0.002400, 0.407300, 0.212300],
            [530, 0.165500, 0.862000, 0.042160],
            [560, 0.594500, 0.995000, 0.003900],
            [590, 1.026300, 0.757000, 0.001100],
            [600, 1.062200, 0.631000, 0.000800],
            [620, 0.854450, 0.381000, 0.000190],
            [630, 0.642400, 0.265000, 0.000050],
            [650, 0.283500, 0.107000, 0.000000],
            [720, 0.002899, 0.001047, 0.000000],
        ]
    )

    XYZ2RGB = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )

    #% Coeficientes
    Coef = (
        CIE1931[select_wavelengths, 1:]
        * (np.ones((3, 1)) * D65[select_wavelengths, 1].T).T
    ).T
    N = np.sum(Coef, axis=1)
    #%  Reproduccion de color usando CIE

    xyz = np.dot(Coef, (imagenes_patron[select_wavelengths, :].T * Pesos_ecu).T).T
    # print(N)
    xyz = xyz / N[1]

    rgb = recorte(np.dot(XYZ2RGB, xyz.T).T)

    im_RGB = np.reshape(rgb, shape_imag)

    return im_RGB


def mejor_combinacion(
    imagenes_patron,
    masks,
    color_check,
    Cant_Image,
    type_error="mean",
    imagen_write="off",
):
    stuff = range(np.shape(imagenes_patron)[0])
    subset = list(itertools.combinations(stuff, Cant_Image))

    min_error = 1000000
    a = 0
    for i, Comb in enumerate(subset):
        if i / len(subset) * 100 > a:
            a += 10
            print(
                "Cant imagenes"
                + str(int(Cant_Image))
                + " Avance:"
                + str("{0:.2f}".format(i / len(subset) * 100))
                + str("%")
            )
        # #%%  Reproduccion de color usando CIE

        im_RGB = ReproduccionCie1931(imagenes_patron, select_wavelengths=Comb)
        # im_Lab= cv2.cvtColor(im_RGB, cv2.COLOR_RGB2LAB)
        errores = Error_de_reproduccion([im_RGB], masks, color_check)

        error = error_funtions(errores, type_error)
        # print(error_media)
        if error < min_error:
            min_error = error
            mejor_comb = Comb
        # fun.imshow('Imagen reproducción CIE 1931',im_RGB)

    #%%  Reproduccion de color usando CIE
    im_RGB = ReproduccionCie1931(imagenes_patron, select_wavelengths=mejor_comb)
    imshow("IR ERGB CIE 1931 im " + str(int(Cant_Image)), im_RGB)
    if imagen_write == "on":
        imwrite(
            "Resultados/Imagenes/IR ERGB CIE 1931 im " + str(int(Cant_Image)) + ".png",
            im_RGB,
        )

    return mejor_comb, min_error


def error_funtions(errores, type_error):
    type_error = type_error.lower()

    if type_error == "mean":
        error = np.mean(errores)

    if type_error == "max":
        error = np.max(errores)

    if type_error == "variance":
        error = np.var(errores)

    if type_error == "mean_for_standard":
        error = np.mean(errores) * np.sqrt(np.var(errores))

    if type_error == "rango":
        error = np.max(errores) - np.min(errores)

    return error
