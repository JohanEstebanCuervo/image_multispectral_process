# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 22:18:54 2021

@author: Johan Cuervo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from methods.formatter_plots import formatter_graph
from color_reproduction_api import ColorReproduction

nombre = "data/CIETABLES.xls"
carpeta_guardado = "results/images/"
hoja = pd.read_excel(nombre, skiprows=4, sheet_name="Table4")

cie = np.array(hoja.iloc[:-1, :4])

# cie XYZ

fig = plt.figure()
axes = fig.subplots()

axes.plot(cie[:, 0], cie[:, 1], color="r")
axes.plot(cie[:, 0], cie[:, 2], color="g")
axes.plot(cie[:, 0], cie[:, 3], color="b")

axes.set_xlabel("$\lambda$ nm")
axes.legend(("X", "Y", "Z"))

formatter_graph(fig, [axes])

fig.savefig(carpeta_guardado + "CIE1931.pdf", format="pdf")

plt.show()

# Iluminante D65


fig = plt.figure()
axes = fig.subplots()

hoja = pd.read_excel(nombre, skiprows=5, sheet_name="Table1")
d65 = np.array(hoja.iloc[:, :4])
axes.plot(d65[:, 0], d65[:, 2], color="black")

axes.set_xlabel("$\lambda$ nm")
axes.legend(("D65",))

formatter_graph(fig, [axes])
fig.savefig(carpeta_guardado + "Illuminant_D65.pdf", format="pdf")

plt.show()


# 7 wav

espectro = np.array([451, 500, 525, 550, 620, 660, 740, 850])

fig = plt.figure()
axes = fig.subplots()

axes.plot(cie[:, 0], cie[:, 1], color="r")
axes.plot(cie[:, 0], cie[:, 2], color="g")
axes.plot(cie[:, 0], cie[:, 3], color="b")

color = ColorReproduction()
color.wavelengths = espectro
cie_wav, d65 = color.charge_cie()
m, n, base = plt.stem(
    cie_wav[:, 0],
    cie_wav[:, 1],
    linefmt="black",
    markerfmt="None",
    use_line_collection=False,
)
plt.setp(base, "linewidth", 0)
m, n, base = plt.stem(
    cie_wav[:, 0],
    cie_wav[:, 2],
    linefmt="black",
    markerfmt="None",
    use_line_collection=False,
)
plt.setp(base, "linewidth", 0)
m, n, base = plt.stem(
    cie_wav[:, 0],
    cie_wav[:, 3],
    linefmt="black",
    markerfmt="None",
    use_line_collection=False,
)
plt.setp(base, "linewidth", 0)

axes.set_xlabel("$\lambda$ nm")
axes.legend(("X", "Y", "Z"))

formatter_graph(fig, [axes])

fig.savefig(carpeta_guardado + "CIE1931_wavs8.pdf", format="pdf")

plt.show()

# 12 wav

espectro = np.array([410, 450, 470, 490, 505, 530, 560, 590, 600, 620, 630, 650])

fig = plt.figure()
axes = fig.subplots()

axes.plot(cie[:, 0], cie[:, 1], color="r")
axes.plot(cie[:, 0], cie[:, 2], color="g")
axes.plot(cie[:, 0], cie[:, 3], color="b")

color = ColorReproduction()
color.wavelengths = espectro
cie_wav, d65 = color.charge_cie()
m, n, base = plt.stem(
    cie_wav[:, 0],
    cie_wav[:, 1],
    linefmt="black",
    markerfmt="None",
    use_line_collection=False,
)
plt.setp(base, "linewidth", 0)
m, n, base = plt.stem(
    cie_wav[:, 0],
    cie_wav[:, 2],
    linefmt="black",
    markerfmt="None",
    use_line_collection=False,
)
plt.setp(base, "linewidth", 0)
m, n, base = plt.stem(
    cie_wav[:, 0],
    cie_wav[:, 3],
    linefmt="black",
    markerfmt="None",
    use_line_collection=False,
)
plt.setp(base, "linewidth", 0)

axes.set_xlabel("$\lambda$ nm")
axes.legend(("X", "Y", "Z"))

formatter_graph(fig, [axes])

fig.savefig(carpeta_guardado + "CIE1931_wavs12.pdf", format="pdf")

plt.show()
