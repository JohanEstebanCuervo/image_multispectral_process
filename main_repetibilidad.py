"""
Programmed by: Johan Esteban Cuervo Chica

Modulo principal de ejemplo para la reproducción y correción de color
en imagenes multiespectrales
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from color_reproduction_api import ColorReproduction
from methods.color_correction import ColorCorrection
from methods.formatter_plots import formatter_graph

# %% Datos entrada

FOLDER = r"imgs\Pruebas de repetibilidad"
NUM_WAVES = 8
SEPARATORS = [r"\_", r" "]  # Si esta entre parentesis puede asignar None

if __name__ == "__main__":
    # %% Inicialización de objetos
    color = ColorReproduction()
    color.separators = SEPARATORS

    color.load_capture(
        r"imgs\Pruebas de repetibilidad\2023_4_23_21_27_50", NUM_WAVES, up_wave=True
    )

    color_correc = ColorCorrection()
    color_correc.color_checker_detection(color.images, imshow=False)

    names = color.wavelengths
    names.insert(0, "capture")
    names.insert(1, "num_mask")
    table_info = pd.DataFrame(columns=names)

    index = 0
    for capture_folder in os.listdir(FOLDER):
        print(capture_folder, end="\r")
        color.load_capture(FOLDER + f"/{capture_folder}", NUM_WAVES, up_wave=True)
        image_m = color.convert_list_images(color.images)

        # values_patch = list(
        #     color_correc.ext_patch(image_m, color_correc.masks[19]).mean(axis=0) / 255
        # )
        for num_mask in range(18, 24):
            values_patch = list(
                color_correc.ext_patch(image_m, color_correc.masks[num_mask])[1750, :]
                / 255
            )
            values_patch.insert(0, capture_folder)
            values_patch.insert(1, num_mask)

            table_info.loc[index] = values_patch
            index += 1

    fig, axes = plt.subplots()
    for num_mask in range(18, 24):
        data_patch = table_info[table_info.num_mask == num_mask]
        min_values = data_patch.min(numeric_only=True).to_numpy()[1:-1]
        max_values = data_patch.max(numeric_only=True).to_numpy()[1:-1]
        mean_values = data_patch.mean(numeric_only=True).to_numpy()[1:-1]

        color_plot = list(np.ones(3) * np.mean(mean_values) * 0.8)
        axes.plot(
            color.wavelengths[:-1],
            max_values,
            linestyle="--",
            color=color_plot,
            label="_nolegend_",
        )
        axes.plot(color.wavelengths[:-1], mean_values, linestyle="-", color=color_plot)
        axes.plot(
            color.wavelengths[:-1],
            min_values,
            linestyle="--",
            color=color_plot,
            label="_nolegend_",
        )

    axes.set_ylim([0, 1])
    axes.set_xlabel("Longitud de onda [nm]")
    axes.set_ylabel("Reflactancia [%]")
    axes.legend(
        ["Blanco", "Neutral 8", "Neutral 6.5", "Neutral 5", "Neutral 3.5", "Negro"]
    )
    formatter_graph(fig, [axes])
    fig.savefig("results/images/repetibilidad.pdf")
    plt.show()
