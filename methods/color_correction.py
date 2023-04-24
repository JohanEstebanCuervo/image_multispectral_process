"""
Programmed By: Johan Esteban Cuervo Chica

Module contains algorithms for color_correction
"""
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras


def root(array: np.ndarray, exp: float) -> np.ndarray:
    """
    Calculate the root array usage np.power(array, 1/exp)

    Args:
        array (np.ndarray): _description_
        exp (_type_): _description_

    Returns:
        np.ndarray: _description_
    """

    return np.power(array, 1 / exp)


class ModelCorrection:
    """
    Model generate ccm_matrix
    """

    def __init__(self) -> None:
        self.__input = None
        self.__output = None
        self.str_input = None
        self.str_output = None
        self.size_model = None
        self.ccm = None
        self.neural_network = None
        self.model = {}

        self.__reg1 = re.compile(r"\([rgbn,\d\.]+\)")
        self.models = {
            "exp": np.exp,
            "ln": np.log,
            "power": np.power,
            "root": root,
            "mult": np.multiply,
        }

        self.bands = ["r", "g", "b", "n"]

        self.inverse = {
            "exp": "ln",
            "ln": "exp",
            "power": "root",
            "root": "power",
        }

    def __charge_arg(self, model, arg):
        args = arg[1:-1].split(",")
        arg_mod = []
        for arg in args:
            if arg in self.bands:
                arg_mod.append({"input": self.bands.index(arg)})
            else:
                arg_mod.append({"value": float(arg)})

        model["args"] = arg_mod

        return model

    def __parser(self, mod_input, type_arg):
        model = []
        input_clean = re.sub(r"\s", "", mod_input).lower()
        args = self.__reg1.findall(input_clean)

        input_functions = input_clean
        for arg in args:
            input_functions = re.sub(re.escape(arg), "", input_functions, 1)

        input_params = input_functions.split(",")

        if type_arg.lower() == "input":
            dict_mod = self.models
        elif type_arg.lower() == "output":
            dict_mod = self.inverse
        else:
            raise ValueError(f"No type {type_arg}")

        for param in input_params:
            if param in dict_mod:
                if type_arg.lower() == "output":
                    param = dict_mod[param]
                mod = {"function": param}
                arg_f = args.pop(0)

                mod = self.__charge_arg(mod, arg_f)

            elif param in self.bands:
                mod = {"input": self.bands.index(param)}

            else:
                raise ValueError(
                    f"Valor no es una banda ni una función conocida: {param}"
                )

            model.append(mod)

        return model

    def compile(self, mod_input: str, output: str) -> None:
        """
        Compila el modelo

        Args:
            mod_input (str): _description_
            output (str): _description_
        """
        self.model = {}

        self.model["input"] = self.__parser(mod_input, "input")
        self.model["output"] = self.__parser(output, "output")

        self.size_model = [len(self.model["input"]) + 1, len(self.model["output"])]

        self.__input = mod_input
        self.__output = output

    def __normalice_vals(self, array: np.ndarray) -> np.ndarray:
        return (array.astype("float") + 1) / 256

    def __procces_matrix(
        self, model: list, image: np.ndarray, constant: bool = True
    ) -> np.ndarray:
        shape_image = np.shape(image)

        image_array = image.reshape((-1, shape_image[-1])).astype("float")

        input_matrix = []

        for mod_input in model:
            type_arg = list(mod_input.keys())[0]

            if type_arg == "function":
                args = mod_input["args"]
                val_arg = []
                for arg in args:
                    if list(arg.keys())[0] == "value":
                        val_arg.append(arg["value"])
                    else:
                        val_arg.append(image_array[:, arg["input"]])

                if len(args) == 2:
                    func = self.models[mod_input["function"]]
                    val_array = func(val_arg[0], val_arg[1])

                elif len(args) == 1:
                    func = self.models[mod_input["function"]]
                    val_array = func(val_arg[0])

            else:
                val_array = image_array[:, mod_input["input"]]

            input_matrix = np.concatenate((input_matrix, val_array), axis=0)
            val_array = None

        if constant:
            input_matrix = np.concatenate(
                (input_matrix, np.ones(len(image_array))), axis=0
            )

        return input_matrix.reshape((-1, len(image_array))).T

    def color_correction(self, rgbn_image) -> np.ndarray:
        size = np.shape(rgbn_image)
        input_arr = self.__procces_matrix(
            self.model["input"], self.__normalice_vals(rgbn_image)
        )

        output_arr = self.ccm @ input_arr.T

        rgb = self.__procces_matrix(self.model["output"], output_arr.T, False)

        im_rgb = np.reshape(rgb, size) * 256 - 1
        im_rgb[np.where(im_rgb > 255)] = 255
        im_rgb[np.where(im_rgb < 0)] = 0

        return im_rgb.astype("uint8")

    def train(self, ideal_values, rgbn_values) -> np.ndarray:
        model_out = self.__parser(self.__output, "input")

        input_arr = self.__procces_matrix(
            self.model["input"], self.__normalice_vals(rgbn_values)
        )
        output_arr = self.__procces_matrix(
            model_out, self.__normalice_vals(ideal_values), constant=False
        )

        pseudo_inv = np.linalg.pinv(input_arr)
        self.ccm = output_arr.T @ pseudo_inv.T

        return self.ccm

    def color_correction_nn(self, rgb_image) -> np.ndarray:
        shape_image = np.shape(rgb_image)
        image_array = rgb_image.reshape((-1, shape_image[-1])).astype("float")

        image_array = image_array / 255

        print("predicción")
        array_out = self.neural_network.predict(image_array, steps=1)

        im_rgb = np.reshape(array_out, shape_image)
        im_rgb[np.where(im_rgb > 255)] = 255
        im_rgb[np.where(im_rgb < 0)] = 0

        return im_rgb.astype("uint8")

    def train_nn(
        self, ideal_values, rgbn_values, epochs=60, batch_size=20
    ) -> keras.models.Sequential:
        samples = len(ideal_values)
        indexs = list(range(samples))
        np.random.shuffle(indexs)
        rgbn_values2 = rgbn_values / 255

        y_train = ideal_values[indexs[: int(samples * 0.75)]]
        x_train = rgbn_values2[indexs[: int(samples * 0.75)]]

        y_val = ideal_values[indexs[int(samples * 0.75) :]]
        x_val = rgbn_values2[indexs[int(samples * 0.75) :]]

        table = np.concatenate((x_train, y_train), axis=1)

        table = pd.DataFrame(
            table, columns=["R_in", "G_in", "B_in", "R_out", "G_out", "B_out"]
        )
        table = table.sort_values(by=["R_out", "G_out", "B_out"])

        table.to_excel("datos_train.xlsx", "datos", engine="xlsxwriter")

        print(f"sizes trian: {np.shape(x_train)} {np.shape(y_train)}")
        print(f"sizes test: {np.shape(x_val)} {np.shape(y_val)}")

        if self.neural_network is None:
            red = keras.models.Sequential(
                [
                    keras.layers.Dense(15, activation="sigmoid", input_shape=(3,)),
                    # Dense(10, activation="sigmoid"),
                    keras.layers.Dense(3, activation="relu"),
                ]
            )
            red.compile(
                optimizer="Adam",
                loss="mean_squared_error",
                metrics=["mean_absolute_error"],
            )
        else:
            red = self.neural_network

        hist = red.fit(
            table[["R_in", "G_in", "B_in"]],
            table[["R_out", "G_out", "B_out"]],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
        )

        red.save("Correction_color_neuronal_red.h5")

        plt.plot(hist.history["loss"])
        plt.plot(hist.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Val"], loc="upper right")
        plt.show()

        self.neural_network = red
        return red

    def __str__(self):
        struct = {
            "input": self.__input,
            "output": self.__output,
            "ccm": (self.ccm.round(3)).tolist(),
        }

        return json.dumps(struct, indent=4)


class ColorCorrection:
    """
    Class con algoritmhs for color correction, read and write CCM

    CCM: Color Correction Matrix.
    """

    def __init__(self) -> None:
        self.image_rgbn = None
        self.masks = None
        self.ideal_color_patch = np.array(
            [
                [116, 81, 67],
                [199, 147, 129],
                [91, 122, 156],
                [90, 108, 64],
                [130, 128, 176],
                [92, 190, 172],
                [224, 124, 47],
                [68, 91, 170],
                [198, 82, 97],
                [94, 58, 106],
                [159, 189, 63],
                [230, 162, 39],
                [34, 63, 147],
                [67, 149, 74],
                [180, 49, 57],
                [238, 198, 32],
                [193, 84, 151],
                [12, 136, 170],
                [243, 238, 243],
                [200, 202, 202],
                [161, 162, 161],
                [120, 121, 120],
                [82, 83, 83],
                [49, 48, 51],
            ]
        )
        self.model = ModelCorrection()
        self.ccm_matrix = {}

    def model_write(self, name: str) -> None:
        """
        Save the CCM file in format .csv
        Args:
            name (str): name file save.
            matrix (np.ndarray): ccm matrix
        """

        with open(name, "wt", encoding="utf8") as file:
            file.write(str(self.model))

    def ccm_read(self, name: str) -> np.ndarray:
        print("hola")

    def load_nn(self, path: str) -> keras.models.Sequential:
        red = keras.models.load_model(path)
        self.model.neural_network = red
        return red

    def create_model(self, input_model, output_model):
        self.model.compile(input_model, output_model)

    def __ideal_color_patch_pixel(self):
        color_ipmask = np.zeros((0, 3))
        for i, mask in enumerate(self.masks):
            num_pixels = np.shape(np.where(mask == 255))[1]

            ideal_mask = np.ones((num_pixels, 3)) * self.ideal_color_patch[i, :]
            color_ipmask = np.concatenate(
                (color_ipmask, ideal_mask), axis=0
            )  # concatena el color ideal de los 24 parches

        return color_ipmask

    def ext_patch(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extrae los valores de un parche de una imagen

        Args:
            image (np.ndarray): Imagen de tamaño (Witdh, Height, Channels)
            mask (np.ndarray): de tamaño (Width, height)

        Raises:
            IndexError: si la mascara no tiene las dimensiones adecuadas
            IndexError: si la imagen no tiene las dimensiones adecuadas
            IndexError: si las dimensiones de la mascara y la imagen no coinciden

        Returns:
            np.ndarray: values masks size (Num_pixels, Channels)
        """
        image = np.array(image)
        mask = np.array(mask)

        size_image = np.shape(image)
        size_mask = np.shape(mask)

        if len(size_mask) != 2:
            raise IndexError(
                f"Tamaño inesperado de mascara se espera matriz de 2 dimensiones tamaño actual: {size_mask}"
            )
        if len(size_image) > 3 or len(size_image) <= 1:
            raise IndexError(
                f"Tamaño inesperado de las imagenes. se espera matriz de 2 o 3 dimensiones tamaño actual: {size_image}"
            )
        if not np.equal(size_image[:2], size_mask[:2]).all():
            raise IndexError(
                f"Tamaño de imagen y de mascaras no coincidentes: {size_image[:2]} != {size_mask[:2]}"
            )

        if len(size_image) == 2:
            return image[np.where(mask == 255)]

        patch_val = np.zeros((len(np.where(mask == 255)[0]), 0))

        for index in range(size_image[-1]):
            channel = image[:, :, index]
            patch = channel[np.where(mask == 255)].reshape((-1, 1))
            patch_val = np.concatenate((patch_val, patch), axis=1)

        return patch_val

    def __ext_patchs(self, rgb_image):
        parches_r = []
        parches_g = []
        parches_b = []

        for mask in self.masks:
            R, G, B = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
            parte = R[np.where(mask == 255)]
            parches_r = np.concatenate((parches_r, parte))
            parte = G[np.where(mask == 255)]
            parches_g = np.concatenate((parches_g, parte))
            parte = B[np.where(mask == 255)]
            parches_b = np.concatenate((parches_b, parte))

        parches_rgb = np.zeros((len(parches_r), 3))
        parches_rgb[:, 0] = parches_r
        parches_rgb[:, 1] = parches_g
        parches_rgb[:, 2] = parches_b

        return parches_rgb

    def train(self, rgbn_image) -> np.ndarray:
        ideal_values = self.__ideal_color_patch_pixel()
        rgb_values = self.__ext_patchs(rgbn_image)

        return self.model.train(ideal_values, rgb_values)

    def train_nn(self, rgb_image, epochs=60, batch_size=20) -> keras.models.Sequential:
        ideal_values = self.__ideal_color_patch_pixel()
        rgb_values = self.__ext_patchs(rgb_image)

        return self.model.train_nn(
            ideal_values, rgb_values, epochs=epochs, batch_size=batch_size
        )

    def color_checker_detection(
        self,
        images: list[np.ndarray],
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
            imshow (bool | str, optional): show images the process algorithm
                options (True, False, 'end'). Defaults to False.
            size_color_checker (tuple[int, int], optional): colorchecker size patchs usually
                (6,4) or (14,10). Defaults to (Automatic detection).

        Returns:
            list[np.ndarray]: list masks patchs colorchecker
        """

        self.masks = color_checker_detection(images, imshow, size_color_checker)

        return self.masks

    def color_correction(self, rgb_image):
        image = self.model.color_correction(rgb_image)

        return image

    def color_correction_nn(self, rgb_image):
        image = self.model.color_correction_nn(rgb_image)

        return image

    def error_lab(self, rgb_imagen):
        error = []
        lab_image = trf.rgb2lab(rgb_imagen)
        imagen = lab_image.reshape(-1, 3)

        ideal_path = trf.rgb2lab(self.ideal_color_patch)

        for i, mask in enumerate(self.masks):
            indices = np.where(
                mask.reshape(
                    -1,
                )
                == 255
            )
            dif = imagen[indices] - ideal_path[i]
            dist_eucl = np.sqrt(np.sum(np.power(dif, 2), axis=1))
            error.append(np.mean(dist_eucl))

        return error


if __name__ == "__main__":
    from color_checker_detection import color_checker_detection
    import transforms as trf

else:
    from .color_checker_detection import color_checker_detection
    from . import transforms as trf
