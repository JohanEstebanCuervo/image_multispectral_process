"""
Programmed by: Johan Esteban Cuervo Chica

"""
import itertools
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


def read_capture(folder: str, listing: list[str]) -> Union[list, np.ndarray, tuple]:
    """
    read a list of images from a folder,
    groups them into an array of  number_images * number_pixels

    Args:
        folder (str): folder path
        listing (list[str]): names images list

    Returns:
        Union[list, np.ndarray, tuple]:list images, matrix the images number_images*numpixels,
        shape origin image
    """
    imagenespatron = []
    images = []
    for name in sorted(listing):
        imagen = cv2.imread(folder + "/" + name, cv2.IMREAD_GRAYSCALE)

        # se convierte la imagen en una fila y se concatena con las demas del espectro
        images.append(imagen)
        imagenespatron = np.concatenate(
            (imagenespatron, np.reshape(imagen, (-1))), axis=0
        )

    shape_imag = np.shape(imagen)

    # se redimensiona a N imagenes multiespectrales * Num_pixels image
    imagenespatron = imagenespatron.reshape(len(listing), -1)

    return images, imagenespatron / 255, shape_imag


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
        imagen1 = np.zeros((np.shape(imagen)[0], np.shape(imagen)[1], 3)).astype(
            "uint8"
        )
        imagen1[:, :, 0] = imagen
        imagen1[:, :, 1] = imagen
        imagen1[:, :, 2] = imagen
        imagen = imagen1

    plt.imshow(imagen, vmin=0, vmax=255)
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


#%% Funciones ccm para una imagen
# Color Correction matriz linear
def CCM_Linear(im_rgb, colorn, masks, shape_imag=(480, 640, 3)):
    entrada = RGB_IN(im_rgb, masks).T
    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = colorn.T / 255

    pseudoinv = np.linalg.pinv(entrada)

    ccm = np.dot(colorn, pseudoinv)
    rgb = np.reshape(im_rgb, (-1, 3)).T
    rgb = np.concatenate((rgb, np.ones((1, np.shape(rgb)[1]))))
    rgb = np.dot(ccm, rgb)
    im_rgb = np.reshape(rgb.T, shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb, ccm


# Color Correction matriz Compound
def CCM_Compound(im_rgb, colorn, masks, shape_imag=(480, 640, 3)):
    entrada = RGB_IN(im_rgb, masks).T
    # entrada_n = N_IN(N, masks)
    # entrada = np.concatenate((entrada,entrada_n))
    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = np.log(colorn.T / 255)

    pseudoinv = np.linalg.pinv(entrada)

    ccm = np.dot(colorn, pseudoinv)
    rgb = np.reshape(im_rgb, (-1, 3)).T
    # rgbn= np.concatenate((rgb,np.reshape(N,(1,-1))))
    rgb = np.concatenate((rgb, np.ones((1, np.shape(rgb)[1]))))

    lnrgb = np.dot(ccm, rgb)
    rgb = np.exp(lnrgb)
    im_rgb = np.reshape(rgb.T, shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb, ccm


# Color Correction matriz Logarithm
def CCM_Logarithm(im_rgb, colorn, masks, shape_imag=(480, 640, 3)):
    entrada = np.log(offset(RGB_IN(im_rgb, masks).T))
    # entrada_n = np.log(offset(N_IN(N, masks)))
    # entrada = np.concatenate((entrada,entrada_n))
    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = colorn.T / 255

    pseudoinv = np.linalg.pinv(entrada)

    ccm = np.dot(colorn, pseudoinv)
    lnrgb = np.log(offset(np.reshape(im_rgb, (-1, 3)).T))
    # lnrgbn = np.concatenate((lnrgb,np.log(offset(np.reshape(N,(1,-1))))))
    lnrgb = np.concatenate((lnrgb, np.ones((1, np.shape(lnrgb)[1]))))

    rgb = np.dot(ccm, lnrgb)
    im_rgb = np.reshape(rgb.T, shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb, ccm


# Color Correction matriz Polynomial Con NIR
def CCM_Polynomial_N(im_rgb, N, colorn, masks):

    entrada = RGB_IN(im_rgb, masks).T
    entrada_n = N_IN(N, masks)
    entrada = np.concatenate((entrada, entrada_n))

    r2 = entrada[0, :] ** 2
    g2 = entrada[1, :] ** 2
    b2 = entrada[2, :] ** 2
    n2 = entrada_n**2
    rg = entrada[0, :] * entrada[1, :]
    rb = entrada[0, :] * entrada[2, :]
    rn = entrada[0, :] * entrada_n
    gb = entrada[1, :] * entrada[2, :]
    gn = entrada[1, :] * entrada_n
    bn = entrada[2, :] * entrada_n

    entrada = np.concatenate((entrada, [r2]))
    entrada = np.concatenate((entrada, [g2]))
    entrada = np.concatenate((entrada, [b2]))
    entrada = np.concatenate((entrada, n2))
    entrada = np.concatenate((entrada, [rg]))
    entrada = np.concatenate((entrada, [rb]))
    entrada = np.concatenate((entrada, rn))
    entrada = np.concatenate((entrada, [gb]))
    entrada = np.concatenate((entrada, gn))
    entrada = np.concatenate((entrada, bn))
    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = colorn.T / 255

    pseudoinv = np.linalg.pinv(entrada)

    entrada_n = np.reshape(N, (1, -1))

    ccm = np.dot(colorn, pseudoinv)
    rgb = np.reshape(im_rgb, (-1, 3)).T
    rgb = np.concatenate((rgb, entrada_n))
    r2 = rgb[0, :] ** 2
    g2 = rgb[1, :] ** 2
    b2 = rgb[2, :] ** 2
    n2 = entrada_n**2
    rg = rgb[0, :] * rgb[1, :]
    rb = rgb[0, :] * rgb[2, :]
    rn = rgb[0, :] * entrada_n
    gb = rgb[1, :] * rgb[2, :]
    gn = rgb[1, :] * entrada_n
    bn = rgb[2, :] * entrada_n

    rgb = np.concatenate((rgb, [r2]))
    rgb = np.concatenate((rgb, [g2]))
    rgb = np.concatenate((rgb, [b2]))
    rgb = np.concatenate((rgb, n2))
    rgb = np.concatenate((rgb, [rg]))
    rgb = np.concatenate((rgb, [rb]))
    rgb = np.concatenate((rgb, rn))
    rgb = np.concatenate((rgb, [gb]))
    rgb = np.concatenate((rgb, gn))
    rgb = np.concatenate((rgb, bn))
    rgb = np.concatenate((rgb, np.ones((1, np.shape(rgb)[1]))))

    rgb = np.dot(ccm, rgb)
    im_rgb = np.reshape(rgb.T, (480, 640, 3))
    im_rgb = recorte(im_rgb)
    return im_rgb, ccm, r2


# Color Correction matriz Polynomial
def CCM_Polynomial(im_rgb, colorn, masks, shape_imag=(480, 640, 3)):

    entrada = RGB_IN(im_rgb, masks).T

    r2 = entrada[0, :] ** 2
    g2 = entrada[1, :] ** 2
    b2 = entrada[2, :] ** 2
    rg = entrada[0, :] * entrada[1, :]
    rb = entrada[0, :] * entrada[2, :]
    gb = entrada[1, :] * entrada[2, :]

    entrada = np.concatenate((entrada, [r2]))
    entrada = np.concatenate((entrada, [g2]))
    entrada = np.concatenate((entrada, [b2]))
    entrada = np.concatenate((entrada, [rg]))
    entrada = np.concatenate((entrada, [rb]))
    entrada = np.concatenate((entrada, [gb]))

    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = colorn.T / 255

    pseudoinv = np.linalg.pinv(entrada)

    ccm = np.dot(colorn, pseudoinv)
    rgb = np.reshape(im_rgb, (-1, 3)).T
    r2 = rgb[0, :] ** 2
    g2 = rgb[1, :] ** 2
    b2 = rgb[2, :] ** 2
    rg = rgb[0, :] * rgb[1, :]
    rb = rgb[0, :] * rgb[2, :]
    gb = rgb[1, :] * rgb[2, :]

    rgb = np.concatenate((rgb, [r2]))
    rgb = np.concatenate((rgb, [g2]))
    rgb = np.concatenate((rgb, [b2]))
    rgb = np.concatenate((rgb, [rg]))
    rgb = np.concatenate((rgb, [rb]))
    rgb = np.concatenate((rgb, [gb]))
    rgb = np.concatenate((rgb, np.ones((1, np.shape(rgb)[1]))))

    rgb = np.dot(ccm, rgb)
    im_rgb = np.reshape(rgb.T, shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb, ccm, r2


#%% FUNCIONES ccm Para multiples imagenes
# Color Correction matriz linear
def CCM_Linear_Train(archivo):
    datatrain = pd.read_csv(archivo, sep=",", names=range(1, 7))
    datatrain = datatrain.to_numpy()
    entrada = datatrain[:, :3].T / 255
    colorn = datatrain[:, 3:]
    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = colorn.T / 255

    pseudoinv = np.linalg.pinv(entrada)
    Ccm_Linear = np.dot(colorn, pseudoinv)

    return Ccm_Linear


# Color Correction matriz Compound
def CCM_Compound_Train(archivo):
    datatrain = pd.read_csv(archivo, sep=",", names=range(1, 7))
    datatrain = datatrain.to_numpy()
    entrada = datatrain[:, :3].T / 255
    colorn = datatrain[:, 3:]
    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = np.log(colorn.T / 255)

    pseudoinv = np.linalg.pinv(entrada)

    Ccm_Compound = np.dot(colorn, pseudoinv)

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

    pseudoinv = np.linalg.pinv(entrada)

    Ccm_Logatirhm = np.dot(colorn, pseudoinv)

    return Ccm_Logatirhm


# Color Correction matriz Polynomial
def CCM_Polynomial_Train(archivo):
    datatrain = pd.read_csv(archivo, sep=",", names=range(1, 7))
    datatrain = datatrain.to_numpy()
    entrada = datatrain[:, :3].T / 255
    colorn = datatrain[:, 3:]

    r2 = entrada[0, :] ** 2
    g2 = entrada[1, :] ** 2
    b2 = entrada[2, :] ** 2
    rg = entrada[0, :] * entrada[1, :]
    rb = entrada[0, :] * entrada[2, :]
    gb = entrada[1, :] * entrada[2, :]

    entrada = np.concatenate((entrada, [r2]))
    entrada = np.concatenate((entrada, [g2]))
    entrada = np.concatenate((entrada, [b2]))
    entrada = np.concatenate((entrada, [rg]))
    entrada = np.concatenate((entrada, [rb]))
    entrada = np.concatenate((entrada, [gb]))

    entrada = np.concatenate((entrada, np.ones((1, np.shape(entrada)[1]))))
    colorn = colorn.T / 255

    pseudoinv = np.linalg.pinv(entrada)

    Ccm_Polynomial = np.dot(colorn, pseudoinv)

    return Ccm_Polynomial


#%% ccm test
# Color Correction matriz linear
def CCM_Linear_Test(im_rgb, ccm):
    shape_imag = np.shape(im_rgb)
    rgb = np.reshape(im_rgb, (-1, 3)).T
    rgb = np.concatenate((rgb, np.ones((1, np.shape(rgb)[1]))))
    rgb = np.dot(ccm, rgb)
    im_rgb = np.reshape(rgb.T, shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb


# Color Correction matriz Compound
def CCM_Compound_Test(im_rgb, ccm):
    shape_imag = np.shape(im_rgb)
    rgb = np.reshape(im_rgb, (-1, 3)).T
    rgb = np.concatenate((rgb, np.ones((1, np.shape(rgb)[1]))))

    lnrgb = np.dot(ccm, rgb)
    rgb = np.exp(lnrgb)
    im_rgb = np.reshape(rgb.T, shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb


# Color Correction matriz Logarithm
def CCM_Logarithm_Test(im_rgb, ccm):
    shape_imag = np.shape(im_rgb)
    lnrgb = np.log(offset(np.reshape(im_rgb, (-1, 3)).T))

    lnrgb = np.concatenate((lnrgb, np.ones((1, np.shape(lnrgb)[1]))))

    rgb = np.dot(ccm, lnrgb)
    im_rgb = np.reshape(rgb.T, shape_imag)
    im_rgb = recorte(im_rgb)
    return im_rgb


# Color Correction matriz Polynomial
def CCM_Polynomial_Test(im_rgb, ccm):
    shape_imag = np.shape(im_rgb)
    rgb = np.reshape(im_rgb, (-1, 3)).T
    r2 = rgb[0, :] ** 2
    g2 = rgb[1, :] ** 2
    b2 = rgb[2, :] ** 2
    rg = rgb[0, :] * rgb[1, :]
    rb = rgb[0, :] * rgb[2, :]
    gb = rgb[1, :] * rgb[2, :]

    rgb = np.concatenate((rgb, [r2]))
    rgb = np.concatenate((rgb, [g2]))
    rgb = np.concatenate((rgb, [b2]))
    rgb = np.concatenate((rgb, [rg]))
    rgb = np.concatenate((rgb, [rb]))
    rgb = np.concatenate((rgb, [gb]))
    rgb = np.concatenate((rgb, np.ones((1, np.shape(rgb)[1]))))

    rgb = np.dot(ccm, rgb)
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


def comparacion_color_check(name, im_rgb, color_check_RGB, masks, folder=""):

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

            parchei = im_rgb[np.where(255 == masks[6 * i + j])] * 255
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

        im_rgb = ReproduccionCie1931(imagenes_patron, select_wavelengths=Comb)
        # im_Lab= cv2.cvtColor(im_rgb, cv2.COLOR_RGB2LAB)
        errores = Error_de_reproduccion([im_rgb], masks, color_check)

        error = error_funtions(errores, type_error)
        # print(error_media)
        if error < min_error:
            min_error = error
            mejor_comb = Comb
        # fun.imshow('Imagen reproducción CIE 1931',im_rgb)

    #%%  Reproduccion de color usando CIE
    im_rgb = ReproduccionCie1931(imagenes_patron, select_wavelengths=mejor_comb)
    imshow("IR ERGB CIE 1931 im " + str(int(Cant_Image)), im_rgb)
    if imagen_write == "on":
        imwrite(
            "Resultados/Imagenes/IR ERGB CIE 1931 im " + str(int(Cant_Image)) + ".png",
            im_rgb,
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
