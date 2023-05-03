import matplotlib.pyplot as plt
import numpy as np
from methods.formatter_plots import formatter_graph
from methods.color_correction import ColorCorrection
from color_reproduction_api import ColorReproduction


FOLDER = r"imgs\2023_4_23_21_28_55"
NAME_SAVE = "spectral_firm_domo.pdf"
NUM_WAVES = 12
SEPARATORS = [r"\_", r" "]  # Si esta entre parentesis puede asignar None
NUM_MASK = 18  # de 0 a 23
IDEAL_VALUE = 243

color_obj = ColorReproduction()
color_obj.separators = SEPARATORS
color_obj.load_capture(FOLDER, NUM_WAVES, up_wave=True)

color_correc = ColorCorrection()
color_correc.color_checker_detection(color_obj.images, imshow="end")


weigths = color_obj.calculate_ecualization(color_correc.masks[NUM_MASK], IDEAL_VALUE)
fig = plt.figure()
axes = fig.subplots()

for index, mask in enumerate(color_correc.masks):
    values = []
    color = color_correc.ideal_color_patch[index] / 255
    for ind_w, image in enumerate(color_obj.images):
        mean = weigths[ind_w] * np.mean(color_correc.ext_patch(image, mask)) / 255
        values.append(mean)

    axes.plot(color_obj.wavelengths, values, color=color)

axes.set_xlabel(r"$\lambda[nm]$")
axes.set_ylabel("Reflactancia")
axes.set_ylim(0, 1)
formatter_graph(fig, [axes])

fig.savefig(f"results/images/{NAME_SAVE}")
plt.show()
