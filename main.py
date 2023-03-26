"""
Programmed by: Johan Esteban Cuervo Chica

Modulo principal de ejemplo para la reproducción y correción de color
en imagenes multiespectrales
"""
from color_reproduction_api import ColorReproduction
from methods.color_checker_detection import color_checker_detection
from methods.color_repro import imshow

folder = "imgs/2023_2_21_15_36"

color = ColorReproduction()
color.separators = [r"\_", r"n"]

color.load_capture(folder, 8, up_wave=True)
print(color.wavelengths)
rgb_im = color.reproduccion_cie_1931(
    select_wavelengths=[451, 500, 525, 550, 620, 660, 740]
)

# imshow("reproducción", rgb_im)

masks = color_checker_detection(color.images, True)
