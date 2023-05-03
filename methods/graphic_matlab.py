import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from formatter_plots import formatter_graph

PATH = "data/todos.fig"

PATH_SAVE = "results/images/"


def gaussmf(x_s, mean=0, sigma=1, scale=1):
    return scale * np.exp(-np.square(x_s - mean) / (2 * sigma**2))


fig = loadmat(PATH, squeeze_me=True, struct_as_record=False)

ax1 = fig["hgS_070000"].children.children

colors = (
    np.array(
        [
            (126, 0, 219),
            (0, 70, 255),
            (0, 169, 255),
            (0, 255, 255),
            (0, 255, 84),
            (94, 255, 0),
            (195, 255, 0),
            (255, 223, 0),
            (255, 190, 0),
            (255, 119, 0),
            (255, 79, 0),
            (255, 0, 0),
        ]
    ).astype("float")
    / 255
)
ax2 = []
for line in ax1:
    if line.type == "graph2d.lineseries":
        ax2.append(line)

fig = plt.figure()
axes = fig.subplots()
for i, line in enumerate(ax2[:-3]):
    x = line.properties.XData
    y = line.properties.YData
    axes.plot(x, y, color=colors[i])

axes.set_xlabel(r"$\lambda[nm]$")
axes.set_ylabel("Intensity")
axes.set_xlim((350, 750))
formatter_graph(fig, [axes])
# plt.legend((nombres[0],nombres[-2],nombres[-1]))

fig.savefig(
    PATH_SAVE + "espectral_comp_intensity12.pdf",
    format="pdf",
)

plt.show()

configs = [
    [5, 410, 1],
    [5, 450, 1],
    [5, 470, 1],
    [5, 490, 1],
    [8, 505, 1],
    [5, 530, 1],
    [20, 560, 1],
    [5, 590, 1],
    [15, 600, 1],
    [5, 620, 1],
    [5, 630, 1],
    [5, 650, 1],
]


fig = plt.figure()
axes = fig.subplots()
x = np.linspace(350, 750, 400)
for i, conf in enumerate(configs):
    y = gaussmf(x, conf[1], conf[0], conf[2])
    axes.plot(x, y, color=colors[i])

axes.set_xlabel(r"$\lambda[nm]$")
axes.set_ylabel("Intensidad Relativa")
axes.set_xlim((350, 750))
# plt.legend((nombres[0],nombres[-2],nombres[-1]))
formatter_graph(fig, [axes])
fig.savefig(
    PATH_SAVE + "espectral_comp_relativity12.pdf",
    format="pdf",
)
plt.show()

fig = plt.figure()
axes = fig.subplots()

for i in [1, 5, 7, 10]:
    line = ax2[i]
    x = line.properties.XData
    y = line.properties.YData
    axes.plot(x, y, color=colors[i])

axes.set_xlabel(r"$\lambda[nm]$")
axes.set_ylabel("Intensidad Relativa")
axes.set_xlim((350, 750))
formatter_graph(fig, [axes])
fig.savefig(
    PATH_SAVE + "espectral_comp_intensity4.pdf",
    format="pdf",
)
plt.show()

fig = plt.figure()
axes = fig.subplots()
x = np.linspace(350, 750, 400)
for i in [1, 5, 7, 10]:
    conf = configs[i]
    y = gaussmf(x, conf[1], conf[0], conf[2])
    axes.plot(x, y, color=colors[i])

axes.set_xlabel(r"$\lambda[nm]$")
axes.set_ylabel("Intensidad")
axes.set_xlim((350, 750))
formatter_graph(fig, [axes])

fig.savefig(
    PATH_SAVE + "espectral_comp_relativity4.pdf",
    format="pdf",
)

plt.show()
