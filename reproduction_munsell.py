"""
Programmed by: Johan Esteban Cuervo Chica

Modulo principal de ejemplo para la reproducción y correción de color
en imagenes multiespectrales
"""
import numpy as np
import cv2
from color_reproduction_api import ColorReproduction
from methods.color_correction import ColorCorrection
from methods.color_repro import imshow

# %% Datos entrada

FOLDER = r"imgs\2023_4_11_16_4"
NUM_WAVES = 8
PATH_RED = "redes_entrenadas/Correction_color_neuronal_red_2.22.h5"
SEPARATORS = [r"\_", r"n"]  # Si esta entre parentesis puede asignar None
NUM_MASK = 18
IDEAL_VALUE = 243

if __name__ == "__main__":
    # %% Inicialización de objetos

    color = ColorReproduction()
    color.separators = SEPARATORS

    color.load_capture(FOLDER, NUM_WAVES, up_wave=True)

    color_correc = ColorCorrection()
    color_correc.load_nn(PATH_RED)

    rgb_im = color.reproduccion_cie_1931()
    cv2.imwrite("results/images/reproduction_munsell.png", np.flip(rgb_im, axis=2))
    imshow("reproducción", rgb_im)

    # %% Reproducción Red
    image_nn = color_correc.color_correction_nn(rgb_im)
    cv2.imwrite("results/images/nn_munsell.png", np.flip(image_nn, axis=2))
    imshow("red neuronal", image_nn)
