"""
Programmed by: Johan Esteban Cuervo Chica

"""
from typing import Union
import numpy as np
import cv2
import matplotlib.pyplot as plt


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
    images = []
    for name in sorted(listing):
        image = cv2.imread(folder + "/" + name, cv2.IMREAD_GRAYSCALE)

        # se convierte la image en una fila y se concatena con las demas del espectro
        images.append(image)

    shape_imag = np.shape(image)

    return images, shape_imag


# funcion para mostrar imagenes con matplotlib  con rango de flotantes (0 a  1)
def imshow(titulo: str, image: np.ndarray) -> None:
    """
    Show Image using matplotlib

    Args:
        titulo (str): title graphic
        image (np.ndarray): iamge
    """
    fig = plt.figure()
    axes = fig.subplots()
    if len(np.shape(image)) == 2:
        imagen1 = np.zeros((np.shape(image)[0], np.shape(image)[1], 3)).astype("uint8")
        imagen1[:, :, 0] = image
        imagen1[:, :, 1] = image
        imagen1[:, :, 2] = image
        image = imagen1

    axes.imshow(image, vmin=0, vmax=255)
    axes.set_title(titulo)
    axes.axis("off")
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)
    plt.show()


def imwrite(path: str, image: np.ndarray) -> None:
    """
    Save image using OpenCV and convert image to BGR

    Args:
        path (str): path_file name
        image (np.ndarray): image
    """
    cv2.imwrite(path, np.flip(image, axis=2))


def comparacion_color_check(name, im_rgb, color_check_RGB, masks, folder=""):
    Grosor = 2

    for i in range(4):
        fila = np.zeros((60, Grosor, 3))
        for j in range(6):
            parchei = np.ones((60, 60, 3)) * color_check_RGB[6 * i + j, :]
            fila = np.concatenate((fila, parchei), axis=1)
            fila = np.concatenate((fila, np.zeros((60, Grosor, 3))), axis=1)

        if i == 0:
            image = np.zeros((Grosor, np.shape(fila)[1], 3))

        image = np.concatenate((image, fila), axis=0)

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

        image = np.concatenate((image, fila), axis=0)
        image = np.concatenate(
            (image, np.zeros((Grosor, np.shape(fila)[1], 3))), axis=0
        )

    imshow(folder + "/Comparación Color_Check - " + name, image.astype(int))
    imwrite(folder + "/Comparacion Color_Check - " + name + ".png", image / 255)
