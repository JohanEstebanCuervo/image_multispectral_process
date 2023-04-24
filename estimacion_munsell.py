import cv2
import numpy as np
import colour
import matplotlib.pyplot as plt
from methods.color_correction import ColorCorrection
from methods.transforms import rgb2xyz, rgb2hsv


def extrac_circles(direction, imshow=False, filt_iter=3):
    image = cv2.imread(direction)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if imshow is True:
        plt.imshow(image, vmin=0, vmax=255)
        plt.show()

    shape_image = np.shape(image)

    # _, image2 = cv2.threshold(image, 245, 255, cv2.THRESH_BINARY)
    _, image2 = cv2.threshold(image, 240, 255, cv2.THRESH_OTSU)
    if imshow is True:
        plt.imshow(image2, vmin=0, vmax=255)
        plt.show()

    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype("uint8")

    image2 = cv2.dilate(image2, kernel, iterations=filt_iter)
    image2 = cv2.erode(image2, kernel, iterations=filt_iter)

    if imshow is True:
        plt.imshow(image2, vmin=0, vmax=255)
        plt.show()

    image2 = cv2.Canny(image2, 100, 200)

    if imshow is True:
        plt.imshow(image2, vmin=0, vmax=255)
        plt.show()

    image2 = cv2.dilate(image2, kernel, iterations=3)

    if imshow is True:
        plt.imshow(image2, vmin=0, vmax=255)
        plt.show()

    circles = cv2.HoughCircles(
        image2,
        cv2.HOUGH_GRADIENT,
        1,
        70,
        param1=50,
        param2=30,
        minRadius=0,
        maxRadius=0,
    )

    if circles is None:
        raise ValueError("No se encontraron Circulos en la imagen")

    circles = np.array(circles[0])

    if imshow is True:
        src = image.copy()
        for i in circles:
            center = (int(i[0]), int(i[1]))
            # circle center
            cv2.circle(src, center, 1, (0, 0, 0), 3)
            # circle outline
            radius = int(i[2] * 0.8)
            cv2.circle(src, center, radius, (0, 0, 0), 3)

        plt.imshow(src, vmin=0, vmax=255)
        plt.show()

    median = np.median(circles, axis=0)[-1]
    circles_filt = circles[np.where(circles[:, -1] <= median * 1.5), :][0]
    if imshow is True:
        src = image.copy()
        for i in circles_filt:
            center = (int(i[0]), int(i[1]))
            # circle center
            cv2.circle(src, center, 1, (0, 0, 0), 3)
            # circle outline
            radius = int(i[2] * 0.8)
            cv2.circle(src, center, radius, (0, 0, 0), 3)

        plt.imshow(src, vmin=0, vmax=255)
        plt.show()

    circles_filt[:, -1] = np.mean(circles_filt[:, -1])

    if imshow is True or imshow == "end":
        src = image.copy()
        for i in circles_filt:
            center = (int(i[0]), int(i[1]))
            # circle center
            cv2.circle(src, center, 1, (0, 0, 0), 3)
            # circle outline
            radius = int(i[2] * 0.4)
            cv2.circle(src, center, radius, (0, 0, 0), 3)

        plt.imshow(src, vmin=0, vmax=255)
        plt.show()

    masks = np.zeros((shape_image[0], shape_image[1], len(circles_filt))).astype(
        "uint8"
    )

    for index, circle in enumerate(circles_filt):
        center = (int(circle[0]), int(circle[1]))
        mask = np.zeros_like(image).astype("uint8")
        cv2.circle(mask, center, 1, (0, 0, 0), 3)
        radius = int(circle[2] * 0.4)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)

        masks[:, :, index] = mask

    return masks, circles_filt


PATH = r"results\images\nn_munsell.png"

masks, circles_mask = extrac_circles(PATH, "end", 10)

imagen = cv2.imread(PATH)

color_obj = ColorCorrection()

colors_rgb = []
for index in range(np.shape(masks)[-1]):
    rgb = color_obj.ext_patch(imagen, masks[:, :, index])

    colors_rgb.append(rgb.mean(axis=0).astype("int"))

colors_xyz = rgb2xyz(colors_rgb)

sum_xyz = colors_xyz.sum(axis=1).reshape((-1, 1))

colors_xyY = colors_xyz / sum_xyz

fig, axes = colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)
axes.scatter(colors_xyY[:, 0], colors_xyY[:, 1])
axes.set_title("")
plt.show()

colors_hsv = rgb2hsv(colors_rgb)

sort_index = np.flip(np.argsort(colors_hsv[:, 0]), axis=0)
print(sort_index)


for val, index in enumerate(sort_index):
    center_c = circles_mask[index, :2].astype("int")

    cv2.putText(
        imagen, str(val + 22), center_c, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5
    )

plt.imshow(imagen, vmin=0, vmax=255)
plt.show()
