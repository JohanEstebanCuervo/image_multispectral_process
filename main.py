"""
Programmed by: Johan Esteban Cuervo Chica

Modulo principal de ejemplo para la reproducción y correción de color
en imagenes multiespectrales
"""
import matplotlib.pyplot as plt
import numpy as np
from color_reproduction_api import ColorReproduction
from methods.color_correction import ColorCorrection
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

color_correc = ColorCorrection()
color_correc.color_checker_detection(color.images, imshow=False)

models = [
    ["r, g, b", "r, g, b"],
    ["ln(r), ln(g), ln(b)", "r, g, b"],
    ["r, g, b", "ln(r), ln(g), ln(b)"],
    [
        "r, g, b, mult(r,g), mult(r,b), mult(g,b), power(r,2), power(g,2), power(b,2)",
        "r, g, b",
    ],
]

name_models = ["linear", "compound", "logarithm", "polynomial"]
errores = []
for mod_input, mod_output in models:
    color_correc.create_model(mod_input, mod_output)

    ccm = color_correc.train(rgb_im)

    linear_correc, error = color_correc.color_correction(rgb_im)

    imshow("Correcion", linear_correc)

    errores.append(error)

print(f"Media errores: {np.mean(errores, axis=1)}")


for error in errores:

    plt.plot(range(1, len(error) + 1), error)

plt.grid()
plt.legend(name_models)
plt.show()
