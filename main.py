"""
Programmed by: Johan Esteban Cuervo Chica

Modulo principal de ejemplo para la reproducción y correción de color
en imagenes multiespectrales
"""
from color_reproduction_api import ColorReproduction
from methods.color_repro import imshow

folder = "imgs/2023_3_20_12_18"

color = ColorReproduction()
color.separators = [r"\_", r"n"]

color.load_capture(folder, 8, up_wave=True)
print(color.wavelengths)
rgb_im = color.reproduccion_cie_1931(
    select_wavelengths=[451, 500, 525, 550, 620, 660, 740]
)

imshow("reproducción", rgb_im)

masks = color.color_checker_detection("end")
