"""
Programmed by: Johan Esteban Cuervo Chica

Modulo principal de ejemplo para la reproducción y correción de color
en imagenes multiespectrales
"""
import matplotlib.pyplot as plt
from color_reproduction_api import ColorReproduction
from methods.color_correction import ColorCorrection
from methods.color_repro import imshow


FOLDER = r"imgs\2023_3_20_12_18"

color = ColorReproduction()
color.separators = [r"\_", r"n"]

color.load_capture(FOLDER, 8, up_wave=True)

color_correc = ColorCorrection()
color_correc.color_checker_detection(color.images, imshow="end")
color.calculate_ecualization(color_correc.masks[18], 243)


rgb_im = color.reproduccion_cie_1931()
error_repro = color_correc.error_lab(rgb_im)
imshow("reproducción", rgb_im)

color_correc.train_nn(rgb_im, epochs=90)

imagen = color_correc.color_correction_nn(rgb_im)
error = color_correc.error_lab(imagen)

imshow("corección red", imagen)

plt.plot(range(1, len(error) + 1), error_repro)
plt.plot(range(1, len(error) + 1), error)
plt.grid()
plt.legend(["reproduction", "neural_network"])
plt.savefig("results/errors/figura3.png")
plt.show()
