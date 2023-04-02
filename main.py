"""
Programmed by: Johan Esteban Cuervo Chica

Modulo principal de ejemplo para la reproducción y correción de color
en imagenes multiespectrales
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from color_reproduction_api import ColorReproduction
from methods.color_correction import ColorCorrection
from methods.color_repro import imshow


folder = "imgs/2023_3_20_12_18"

color = ColorReproduction()
color.separators = [r"\_", r"n"]

color.load_capture(folder, 8, up_wave=True)

color_correc = ColorCorrection()
color_correc.color_checker_detection(color.images, imshow="end")

print(color.calculate_ecualization(color_correc.masks[18], 243))
print(color.wavelengths)
rgb_im = color.reproduccion_cie_1931(
    select_wavelengths=[451, 500, 525, 550, 620, 660, 740]
)
cv2.imwrite("results/images/reproduction.png", np.flip(rgb_im, axis=2))
imshow("reproducción", rgb_im)


models = [
    ["r, g, b", "r, g, b"],
    ["ln(r), ln(g), ln(b)", "r, g, b"],
    ["r, g, b", "ln(r), ln(g), ln(b)"],
    [
        "r, g, b, mult(r,g), mult(r,b), mult(g,b), power(r,2), power(g,2), power(b,2)",
        "r, g, b",
    ],
    [
        "r, g, b, mult(r,g), mult(r,b), mult(g,b), power(r,2), power(g,2), power(b,2), ln(r), ln(g), ln(b)",
        "r, g, b",
    ],
    [
        "r, g, b, mult(r,g), mult(r,b), mult(g,b), ln(r), ln(g), ln(b)",
        "r, g, b",
    ],
]

name_models = [
    "reproduction",
    "linear",
    "compound",
    "logarithm",
    "polynomial",
    "superpolynomial",
    "nopowerpolynomial",
]
errores = []
index = 1

error = color_correc.error_lab(rgb_im)
errores.append(error)

for mod_input, mod_output in models:
    color_correc.create_model(mod_input, mod_output)

    ccm = color_correc.train(rgb_im)

    linear_correc, error = color_correc.color_correction(rgb_im)

    cv2.imwrite(
        f"results/images/{name_models[index]}.png", np.flip(linear_correc, axis=2)
    )
    # imshow("Correcion", linear_correc)

    color_correc.model_write(f"results/models/{name_models[index]}.json")
    index += 1
    errores.append(error)

print(f"Media errores: {np.mean(errores, axis=1)}")


for error in errores[:-2]:

    plt.plot(range(1, len(error) + 1), error)

plt.grid()
plt.legend(name_models)
plt.savefig("results/errors/graphic_erros.png")
plt.show()

df = pd.DataFrame(errores, columns=range(1, 25), index=name_models)
df.to_excel("results/errors/errores.xlsx", "errores", engine="xlsxwriter")
