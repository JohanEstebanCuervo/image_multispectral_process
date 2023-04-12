"""
Programmed by: Johan Esteban Cuervo Chica

Module contain algorithms for detection the colorchecker
"""
import numpy as np
import cv2


def size_squares(contours: list) -> tuple[float, float]:
    """
    computes an approximation of the size of the colorchecker boxes,
    from a list of outlines of an image. Filtering the outlines with
    4 sides and a square look

    Args:
        contours (list): contours image

    Returns:
        tuple[float, float]: size upper, size lower
    """
    lista = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        _, _, width, height = cv2.boundingRect(approx)
        if len(approx) == 4 and width > 10 and height > 10:
            if 1.2 > width / height > 0.8:
                lista = np.append(lista, (width, height))
    promedio = np.mean(lista)
    return promedio * 1.1, promedio * 0.9


def contours_images(
    images_list: list, imshow: bool = False, erode_iter: int = 2
) -> list:
    """
    calculates the contours of the image list after applying otsu
    binarization and applying an erosion "erode_iterations (default 2)"
    iterations.

    Args:
        images_list (list): list images multispectral
        imshow (bool, optional): show images binarization usage matplotlib. Defaults to False.
        erode_iter (int, optional): number of iterations in which the erosion of the binarized
        image is applied. Defaults to 2.

    Returns:
        list: contours images.
    """
    contours = []

    for i, imagen in enumerate(images_list):
        _, imagen_bin = cv2.threshold(imagen, 0, 255, cv2.THRESH_OTSU)
        imagen_bin = cv2.erode(imagen_bin, None, iterations=erode_iter)

        if imshow is True:
            func.imshow(f" binary image {i + 1}", imagen_bin)

        list_contours, _ = cv2.findContours(
            imagen_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )  # OpenCV
        if list_contours:
            contours.extend(list_contours)

    return contours


def filter_squeare_cont(contours: list) -> tuple[list, list]:
    """
    Filters out the square outlines that occur in the greatest number
    and have the same size

    Args:
        contours (list): list_contours

    Returns:
        tuple[list, list]: square contours filter, number edges contours
    """
    maximo, minimo = size_squares(contours)
    contornos_cua = []
    number_edges = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        _, _, width, height = cv2.boundingRect(approx)
        if maximo > width > minimo and maximo > height > minimo and len(approx) < 10:
            contornos_cua.append(contour)
            number_edges.append(len(approx))

    return contornos_cua, number_edges


def groups_squeares(centers: list, max_dist: int = 10) -> list[list]:
    """
    groups all the centers that are at a distance less than 'max_dist'.
    creating index groups.

    Args:
        centers (list): list centers squares
        max_dist (int, optional): distance to group. Defaults to 10.

    Returns:
        list[list]: list of groups
    """
    groups = []
    seleccionados = []

    for i, center in enumerate(centers):
        if i not in seleccionados:
            seleccionados.append(i)
            group = [i]
            for j, center2 in enumerate(centers[i + 1 :], i + 1):
                if j not in seleccionados:
                    dist = np.sqrt(np.sum((center - center2) ** 2))
                    if dist < max_dist:
                        seleccionados.append(j)
                        group.append(j)

            groups.append(group)

    return groups


def calculate_centers(contours_squares: list) -> list:
    """
    computes the center of the squares patch colorchecker.

    Args:
        contours_squares (list): contours squares.

    Returns:
        list: centers list.
    """
    centers = []
    centers_res = []

    for contour in contours_squares:
        maxi = np.max(contour, axis=0)
        mini = np.min(contour, axis=0)
        prom = np.mean([maxi, mini], axis=0)[0]
        centers.append(prom)

    groups = groups_squeares(centers)
    centers = np.array(centers)

    for group in groups:
        center = np.mean(centers[group], axis=0)
        centers_res.append(center)

    return centers_res


def angle_size_estimation(contours_squares: list, edges: list) -> tuple[int, int]:
    """
    estimates the size and angle of inclination of the colorchecker patches

    Args:
        contours_squares (list): list of contours
        edges (list): number of edges of the contours

    Returns:
        tuple[int, int]: angle (rad), size edge square
    """
    ang = 0
    size = 0
    contours_filt = []
    for index, edge in enumerate(edges):
        if edge == 4:
            contours_filt.append(contours_squares[index])

    for contour in contours_filt:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        _, tam, ang2 = cv2.minAreaRect(approx)
        size += np.sum(tam)
        if ang2 > 80:
            ang2 = 90 - ang2
        ang += ang2

    ang /= len(contours_filt)
    size /= 2 * len(contours_filt)
    if ang > 45:
        ang -= 90
    return ang * np.pi / 180, size


def angle_regression(
    centers: list, orientation: int, size: tuple[int, int] = None
) -> float:
    """
    estimates the angle of inclination of the colorchecker
    according to a series of central points.

    Args:
        centers (list): centers points
        orientation (int): estimate orientation colorchecker horizontal: 0, vertical: 1
        size (tuple[int, int], optional): colorchecker size patchs usually [6,4] or [14,10].
            Defaults to [6,4].

    Returns:
        float: _description_
    """
    if size is None:
        size = np.array([6, 4])

    else:
        size = np.array(size)

    if orientation == 1:
        points_y = centers[: size[0], 1]
        points_x = centers[: size[0], 0]
    if orientation == 0:
        points_y = centers[: size[0], 0]
        points_x = centers[: size[0], 1]

    pendiente = interpolate.linear_regression(points_x, points_y)[0]
    angle = np.arctan(pendiente)  # *180/np.pi
    return angle


def interpolate_centers(
    centers: list, angle: float, size: tuple[int, int] = None
) -> tuple[list, float, int]:
    """
    fill in the missing centers in the colorchecker according to the
    size of the colorchecker.

    Args:
        centers (list): list centers
        angle (float): angle rotation colorchecker (rad)
        size (tuple[int, int], optional): colorchecker size patchs usually [6,4] or [14,10].
            Defaults to [6,4].

    Returns:
        tuple[list, float, int]: new_centers, error estimated, orientation colorchecker:
        0: horizontal, 1: vertical.
    """

    error = 0

    if size is None:
        size = np.array([6, 4])

    else:
        size = np.array(size)

    matrix_rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    centers_tras = np.copy(centers)
    centers_tras = np.dot(matrix_rotation, centers_tras.T).T

    centers_tras = centers_tras[np.argsort(centers_tras[:, 1]).T]
    maxi, mini = np.max(centers_tras, axis=0), np.min(centers_tras, axis=0)
    spam = maxi - mini
    if spam[0] > spam[1]:
        spam = np.divide(spam, size - 1)
        orientation = 1
    else:
        spam = np.divide(spam, size.flip() - 1)
        orientation = 0

    centers_tras = np.divide(centers_tras - mini, spam)
    centers_int = np.round(centers_tras).astype(int)
    missing = []
    for i in range(int(np.max(centers_int[:, 0]) + 1)):
        for j in range(int(np.max(centers_int[:, 1]) + 1)):
            bandera = 0
            center = np.array([i, j])
            for k in centers_int:
                if np.array_equal(k, center) is True:
                    bandera = 1
                    continue
            if bandera == 0:
                missing.append(center)

    try:
        missing_tras = []
        for center_mis in missing:
            # regresion en direccion 1
            centers = []
            for i, centro_list in enumerate(centers_int):
                if center_mis[0] == centro_list[0]:
                    centers.append(centers_tras[i])

            centers = np.array(centers)
            slope1, inter1 = interpolate.linear_regression(centers[:, 0], centers[:, 1])

            # regresion en direccion 2
            centers = []
            for i, centro_list in enumerate(centers_int):
                if center_mis[1] == centro_list[1]:
                    centers.append(centers_tras[i])
            centers = np.array(centers)

            slope2, inter2 = interpolate.linear_regression(centers[:, 0], centers[:, 1])

            pos_x = (inter2 - inter1) / (slope1 - slope2)
            pos_y = slope1 * pos_x + inter1

            missing_tras.append([pos_x, pos_y])

        missing_tras = np.array(missing_tras)
        centers_tras = np.concatenate((centers_tras, missing_tras))
    except IndexError as error:
        print(f"ERROR!: {error}")
        error = np.max(np.abs(centers_tras - centers_int)) * np.max(spam)
        print("error estimado de posicion: " + str(error))
        centers_tras = np.concatenate((centers_tras, missing))

    centers_tras = centers_tras[np.argsort(centers_tras[:, orientation]).T]

    for i in range(size[1]):
        parte = np.copy(centers_tras[i * size[0] : i * size[0] + size[0], :])
        parte = parte[np.argsort(parte[:, 1 - orientation]).T]
        centers_tras[i * size[0] : i * size[0] + size[0], :] = parte

    centers_tras = centers_tras * spam + mini
    centers = np.dot(np.linalg.inv(matrix_rotation), centers_tras.T).T

    return centers, error, orientation


def generate_masks(
    centers: list,
    size_image: tuple[int, int],
    angle: float,
    size_square: float,
    imshow: bool | str = False,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    generates the masks of the colochecker patches

    Args:
        centers (list): list centers
        size_image (tuple[int, int]): image size
        angle (float): angle rotation colorchecker
        size_square (float): size edge square
        imshow (bool | str, optional): show images the process algorithm
            options (True, False, "end"). Defaults to False.
    Returns:
        tuple[list[np.ndarray], np.ndarray]: list mask arrays, number mask image
    """
    masks = []

    mask_numbers = np.zeros(size_image).astype("int")

    for i, center in enumerate(centers):
        mask = np.zeros((size_image), dtype="uint8")
        point1 = center - round(size_square * 0.9 / 2)
        point2 = center + round(size_square * 0.9 / 2)
        cv2.rectangle(mask, point1, point2, (255, 255, 255), -1)
        mask = transforms.rotate_image(mask, angle, np.flip(center))

        pos_init = np.array(center + [-size_square // 2.4, size_square // 2.4]).astype(
            "int"
        )
        if imshow is True or str(imshow).lower() == "end":
            mask_numbers += mask
            cv2.putText(
                mask_numbers,
                str(i + 1),
                pos_init,
                cv2.FONT_HERSHEY_TRIPLEX,
                size_square / 60,
                (1, 1, 1),
            )
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        masks.append(mask)
    return masks, mask_numbers


def sorted_centers(
    centers: list,
    angle: float,
    size_square: float,
    images: list[np.ndarray],
    size: tuple[int, int] = None,
) -> list:
    """
    Organize centers for classical colorchecker 24 patches. patch number 19
    is white. patch number 1 is  brown

    Args:
        centers (list): list centers
        angle (float): angle rotation colorchecker
        size_square (float): size edge square patch
        images (list[np.ndarray]): list of images
        size (tuple[int, int], optional): colorchecker size patchs usually [6,4] or [14,10]
        (FUTURE USE ONLY WORKS FOR CLASSIC COLORCHECKER). Defaults to is (6, 4).

    Returns:
        list: list centers sorted
    """
    if size is None:
        size = np.array([6, 4])

    else:
        size = np.array(size)

    mascaras_iniciales = [0, 5, 18, 23]
    sums = np.zeros(4)
    size_image = np.shape(images[0])

    for i, pos_centro in enumerate(mascaras_iniciales):
        mask = np.zeros(size_image, dtype="uint8")
        point1 = centers[pos_centro] - round(size_square * 0.9 / 2)
        point2 = centers[pos_centro] + round(size_square * 0.9 / 2)

        cv2.rectangle(mask, point1, point2, (255, 255, 255), -1)
        mask = transforms.rotate_image(mask, -angle, np.flip(centers[pos_centro]))
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

        number_pixels = len(np.where(mask == 255)[0])
        for image in images:
            sums[i] = sums[i] + np.sum(image[np.where(mask == 255)]) / number_pixels

    print(f"size mask {number_pixels} pixels")
    argmax = np.argmax(sums)

    if argmax == 0:
        centers2 = np.zeros((1, 2), dtype="int")
        for i in np.flip(range(size[1])):
            parte = centers[size[0] * i : size[0] * i + size[0], :]
            centers2 = np.concatenate((centers2, parte))
        centers = centers2[1:, :]
    elif argmax == 1:
        centers = np.flip(centers, axis=0)
    elif argmax == 3:
        for i in np.flip(range(size[1])):
            parte = np.flip(centers[size[0] * i : size[0] * i + size[0], :], axis=0)
            centers[size[0] * i : size[0] * i + size[0], :] = parte

    return centers


def color_checker_detection(
    images_list: list[np.ndarray],
    imshow: bool | str = False,
    size_color_checker: tuple[int, int] = None,
) -> list[np.ndarray]:
    """
    Generate mask for colorchecker classical and digital sg. T

    This function uses classic image processing, which is why it
    can be very unstable when faced with variations in position,
    saturation and size of the card. Use with caution. It may not
    work most of the time.

    Args:
        images_list (list[np.ndarray]): list images multispectral
        imshow (bool | str, optional): show images the process algorithm
            options (True, False, 'end'). Defaults to False.
        size_color_checker (tuple[int, int], optional): colorchecker size patchs usually
            (6,4) or (14,10). Defaults to (Automatic detection).

    Returns:
        list[np.ndarray]: list masks patchs colorchecker
    """
    image = images_list[0]
    if imshow is True:
        func.imshow("imagen", image.astype("uint8"))

    size_image = np.shape(image)
    contours = contours_images(images_list, imshow)
    contours, number_edges = filter_squeare_cont(contours)

    if imshow is True:
        imagen = np.zeros(size_image, dtype="uint8")
        contours = np.array(contours, dtype=object)
        cv2.drawContours(imagen, contours, -1, (255, 255, 255), 2)
        func.imshow("contours", imagen)

    centers = calculate_centers(contours)

    print("Squares Detected: " + str(len(centers)))
    if size_color_checker is None:
        if len(centers) < 25:
            size_color_checker = (6, 4)
            print("Color checker estimate classical size (6,4) patches")

        else:
            size_color_checker = (14, 10)
            print("Color checker estimate digital sg size (14,10) patches")

    angulo_est, size_square = angle_size_estimation(contours, number_edges)
    centers_int, _, orientation = interpolate_centers(
        centers, -angulo_est, size_color_checker
    )

    angle = angle_regression(centers_int, orientation, size_color_checker)
    if angle * angulo_est < 0:
        angle = -angle

    print(f"angle estimate contours: {angle * 180 / np.pi}")
    print(f"angle estimate centers:  {angle * 180 / np.pi}")

    centers_int = centers_int.astype(int)
    centers_org = sorted_centers(
        centers_int, angle, size_square, images_list, size_color_checker
    )

    masks, mask_number = generate_masks(
        centers_org, size_image, angle, size_square, imshow
    )

    if imshow is True:
        imagen = np.zeros(size_image, dtype="uint8")
        cv2.drawContours(imagen, contours, -1, (255, 255, 255), 2)
        lista = [centers_int[:, 1], centers_int[:, 0]]
        imagen[tuple(lista)] = 255
        func.imshow("contornos filtradose interpolación de centros", imagen)
        func.imshow("masks", mask_number)

    if imshow is True or str(imshow).lower() == "end":

        image[np.where(mask_number == 255)] = 255
        image[np.where(mask_number == 1)] = 1
        func.imshow("imagen con masks", image)

    return masks


if __name__ == "__main__":
    import interpolate
    import transforms
    import color_repro as func
else:
    from . import interpolate
    from . import transforms
    from . import color_repro as func
