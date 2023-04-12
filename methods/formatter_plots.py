import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 10

# plt.style.use("seaborn-whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def formatter_graph(fig: object, axes_list: list) -> None:
    """
    Formatear Grafica de matplotlib a un estilo especifico

    Args:
        fig (object): _description_
        axes_list (list): _description_
    """
    fig.set_size_inches(4, 3)
    fig.dpi = 300
    fig.set_layout_engine("constrained")

    for axes in axes_list:
        axes.grid(True)
