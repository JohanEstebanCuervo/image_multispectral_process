import cv2
import numpy as np
import colour
import matplotlib.pyplot as plt
from color_reproduction_api import ColorReproduction
from methods.color_correction import ColorCorrection
from methods.transforms import rgb2xyz, rgb2hsv

FOLDER = r"imgs\munsell_3"
NUM_WAVES = 8
PATH_RED = "redes_entrenadas/Correction_color_neuronal_red_2.22.h5"
SEPARATORS = [r"\_", r" "]  # Si esta entre parentesis puede asignar None
NUM_MASK = 18
IDEAL_VALUE = 243
INIT_VALUE = 1
INIT_VALUE_PATCH = 0  # 36  # For no correction


def generate_circle_kernel(size: int = 10):
    """
    Genera un kernel circular para operaciones morfologicas
    Args:
        size (int, optional): Tamaño del kernel. Defaults to 10.
    """
    kernel = np.zeros((size, size)).astype("uint8")

    radius = (size - 1) / 2
    radius_c = radius
    if size % 2 == 0:
        radius_c += 0.5

    for row in range(size):
        for column in range(size):
            calc_r = np.sqrt((row - radius) ** 2 + (column - radius) ** 2)
            if calc_r <= radius_c:
                kernel[row, column] = 1

    return kernel


def extrac_circles(
    image,
    imshow=False,
    stimate_size: int = 11,
    invert_im: bool = False,
    circle_precision: float = 15 / 100,
):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if imshow is True:
        plt.imshow(image, vmin=0, vmax=255)
        plt.show()

    shape_image = np.shape(image)

    if invert_im:
        _, image2 = cv2.threshold(
            image, 128, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV
        )
    else:
        _, image2 = cv2.threshold(image, 128, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    if imshow is True:
        plt.imshow(image2, vmin=0, vmax=255)
        plt.show()

    kernel_c = generate_circle_kernel(stimate_size)

    image2 = cv2.erode(image2, kernel_c, iterations=1)
    image2 = cv2.dilate(image2, kernel_c, iterations=1)

    if imshow is True:
        plt.imshow(image2, vmin=0, vmax=255)
        plt.show()

    _, _, stats, centers = cv2.connectedComponentsWithStats(image2, 1, cv2.CV_8S)
    indexs_circles = []
    low_limit = (np.pi / 4) * (1 - circle_precision)
    max_limit = (np.pi / 4) * (1 + circle_precision)
    low_relation = 1 - circle_precision
    max_relation = 1 + circle_precision
    print(f"limits: {low_limit}, {max_limit}")
    radius = 0
    size_rects = []
    for index, stat in enumerate(stats):
        relation_c = stat[4] / (stat[2] * stat[3])
        relation_aspect = round(stat[2] / stat[3], 1)
        size_rects.append(np.mean(stat[2:4]))
        if (
            relation_aspect >= low_relation
            and relation_aspect <= max_relation
            and relation_c >= low_limit
            and relation_c <= max_limit
        ):
            indexs_circles.append(index)
            radius += stat[2] + stat[3]
    radius /= round((4 * len(indexs_circles)))
    print(f" radio: {radius}")
    circles = []
    low_r = 2 * radius * (0.6)
    max_r = 2 * radius * (1 + circle_precision)
    print(f"limits_circles: {low_r},{max_r}")
    for index in indexs_circles:
        center = centers[index]
        size_circle = size_rects[index]

        if size_circle >= low_r and size_circle <= max_r:
            circles.append([round(center[0]), round(center[1]), radius * 0.5])

    print(f"Total circulos: {len(circles)}")
    if imshow:
        src = image.copy()
        for i in circles:
            center = (i[0], i[1])
            # circle center
            cv2.circle(src, center, 1, (0, 0, 0), 3)
            # circle outline
            radius = int(i[2])
            cv2.circle(src, center, radius, (0, 0, 0), 3)

        plt.imshow(src, vmin=0, vmax=255)
        plt.show()

    masks = np.zeros((shape_image[0], shape_image[1], len(circles))).astype("uint8")

    for index, circle in enumerate(circles):
        center = (int(circle[0]), int(circle[1]))
        mask = np.zeros_like(image).astype("uint8")
        cv2.circle(mask, center, 1, (0, 0, 0), 3)
        radius = int(circle[2] * 0.4)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)

        masks[:, :, index] = mask

    return masks, np.array(circles)


color = ColorReproduction()
color.separators = SEPARATORS

color.load_folder_capture(FOLDER, NUM_WAVES, up_wave=True)

color_correc = ColorCorrection()
color_correc.load_nn(PATH_RED)

xyz_im = color.reproduccion_cie_1931(output_color_space="XYZ")
rgb_im = color.reproduccion_cie_1931(output_color_space="RGB")

masks, circles_mask = extrac_circles(
    image=rgb_im,
    imshow=True,
    stimate_size=10,
    invert_im=False,
    circle_precision=12 / 100,
)

imagen = xyz_im

color_obj = ColorCorrection()
fig, axes = colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)

colors_rgb = []
for index in range(np.shape(masks)[-1]):
    rgb = color_obj.ext_patch(imagen, masks[:, :, index])

    colors_rgb.append(rgb.reshape(-1, 3))

color = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [200, 0, 0],
    [0, 200, 0],
    [0, 0, 200],
    [200, 200, 0],
    [0, 200, 200],
    [200, 0, 200],
    [50, 0, 0],
    [0, 50, 0],
    [0, 0, 50],
    [50, 50, 0],
    [0, 50, 50],
    [50, 0, 50],
    [255, 255, 255],
    [128, 128, 128],
    [0, 0, 0],
]
color_append = color.copy()

for _ in range(4):
    color.extend(color_append)

color = (np.array(color) / 255).tolist()

for index, patch in enumerate(colors_rgb):
    colors_xyz = patch

    sum_xyz = colors_xyz.sum(axis=1).reshape((-1, 1))

    colors_xyY = colors_xyz / sum_xyz

    axes.scatter(colors_xyY[:, 0], colors_xyY[:, 1], color=color[index])
    axes.set_title("")
plt.show()

colors_rgb = []
plt.imshow(rgb_im, vmin=0, vmax=255)
plt.axis("off")
plt.title("Reproduction")
plt.show()

imagen = color_correc.color_correction_nn(rgb_im)
for index in range(np.shape(masks)[-1]):
    rgb = color_obj.ext_patch(imagen, masks[:, :, index])

    colors_rgb.append(rgb.mean(axis=0).astype("int"))

colors_hsv = rgb2hsv(colors_rgb)

sort_index = np.argsort(colors_hsv[:, 0])
print(sort_index)

print(f"tamaño: {np.shape(imagen)}")
print(f"type: {type(imagen)}")

end_value = INIT_VALUE + len(circles_mask) - 1
for val, index in enumerate(sort_index):
    center_c = circles_mask[index, :2].astype("int")

    value = val + INIT_VALUE + INIT_VALUE_PATCH - 1

    if value > end_value:
        value -= end_value

    cv2.putText(
        imagen,
        str(value),
        center_c,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
    )

plt.imshow(imagen, vmin=0, vmax=255)
plt.show()
64, 60, 75
