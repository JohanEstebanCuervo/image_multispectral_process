"""
Este modulo define una clase generica para imagenes multiespectrales.
En la cual se podra cargar y guardar dichas imagenes en formato .pickle
nativo de python.


Ademas proporciona una representación generica de dichas imagenes.
"""
import pickle


class MultiSpectralImage:
    """
    Class MultiEspectral Image
    """

    def __init__(self, path: str = None) -> None:
        self.images = None
        self.wavelengths = None
        self.masks = None
        self.log = None
        self.ecualization_weigths = None

        if path is not None:
            self.load_image(path)

    def save_image(self, path: str) -> None:
        """
        Save MultiEspectralCapture

        Args:
            path (str): path file save capture
        """
        with open(path + ".micpy", "wb") as file:
            pickle.dump(self, file)

    def load_image(self, path: str) -> None:
        """
        Load MultiSpectral Capture
        """
        with open(path, "rb") as file:
            object_im = pickle.load(file)

        if isinstance(object_im, MultiSpectralImage):
            self.images = object_im.images
            self.wavelengths = object_im.wavelengths
            self.masks = object_im.masks
            self.log = object_im.log
            self.ecualization_weigths = object_im.ecualization_weigths

        else:
            raise TypeError(
                f"Object read no type MultiEspectralImage. Type {type(object_im)}"
            )
