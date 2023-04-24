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
from methods.formatter_plots import formatter_graph

# %% Datos entrada

FOLDER = r"imgs\2023_3_20_12_18"
NUM_WAVES = 8
PATH_RED = "redes_entrenadas/Correction_color_neuronal_red_2.22.h5"
SEPARATORS = [r"\_", r"n"]  # Si esta entre parentesis puede asignar None
NUM_MASK = 18
IDEAL_VALUE = 243


def graphic_errors(models: list[str], name_fig: str) -> None:
    """
    Plotea una lista de modelos dados

    Args:
        models (list[str]): Lista de modelos
        name_fig (str): nombre para guarda figura
    """
    fig, axes = plt.subplots()

    for index, name_model in enumerate(name_models):
        if name_model in models:
            error = errores[index]
            axes.plot(range(1, len(error) + 1), error)

    axes.legend(models, loc="upper right")
    axes.set_xlabel("Número de parche")
    axes.set_ylabel("$\Delta E_{Lab}$")
    formatter_graph(fig, [axes])
    fig.savefig(f"results/errors/{name_fig}.pdf")
    fig.show()
    plt.waitforbuttonpress()


if __name__ == "__main__":
    # %% Inicialización de objetos

    color = ColorReproduction()
    color.separators = SEPARATORS

    color.load_capture(FOLDER, NUM_WAVES, up_wave=True)

    color_correc = ColorCorrection()
    color_correc.color_checker_detection(color.images, imshow="end")
    color_correc.load_nn(PATH_RED)
    color.calculate_ecualization(color_correc.masks[NUM_MASK], IDEAL_VALUE)

    rgb_im = color.reproduccion_cie_1931()
    cv2.imwrite("results/images/reproduction.png", np.flip(rgb_im, axis=2))
    imshow("reproducción", rgb_im)

    # %% Modelos de correción
    models_ccm = [
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
        "Reproducción",
        "Lineal",
        "Compuesto",
        "Logarítmica",
        "Polinomial",
        "Superpolynomial",
        "Nopowerpolynomial",
        "Red Neuronal",
    ]

    errores = []
    index = 1

    # %% Error de reproducción
    error = color_correc.error_lab(rgb_im)
    errores.append(error)

    # %% Error y correcion de modelos ccm
    for mod_input, mod_output in models_ccm:
        color_correc.create_model(mod_input, mod_output)

        ccm = color_correc.train(rgb_im)

        ccm_correc = color_correc.color_correction(rgb_im)
        error = color_correc.error_lab(ccm_correc)

        cv2.imwrite(
            f"results/images/{name_models[index]}.png", np.flip(ccm_correc, axis=2)
        )
        # imshow("Correcion", linear_correc)

        color_correc.model_write(f"results/models/{name_models[index]}.json")
        index += 1
        errores.append(error)

    # %% Reproducción Red
    image_nn = color_correc.color_correction_nn(rgb_im)
    cv2.imwrite("results/images/neural_network.png", np.flip(image_nn, axis=2))
    error = color_correc.error_lab(image_nn)
    errores.append(error)

    # %% Guardado de errores y graficas
    print(f"Media errores: {np.mean(errores, axis=1)}")

    df = pd.DataFrame(errores, columns=range(1, 25), index=name_models)
    df.to_excel("results/errors/errores.xlsx", "errores", engine="xlsxwriter")

    graphic_errors(
        [
            "Reproducción",
            "Lineal",
        ],
        "figura1",
    )
    graphic_errors(
        [
            "Reproducción",
            "Compuesto",
            "Logarítmica",
            "Polinomial",
        ],
        "figura2",
    )
    graphic_errors(
        [
            "Reproducción",
            "Red Neuronal",
        ],
        "figura3",
    )
    graphic_errors(
        [
            "Reproducción",
            "Polinomial",
            "Red Neuronal",
        ],
        "figura4",
    )
