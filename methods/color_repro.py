"""
Programmed by: Johan Esteban Cuervo Chica

"""
import itertools
from typing import Union

import numpy as np
import cv2
import matplotlib.pyplot as plt

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
