from color_reproduction_api import ColorReproduction
import cv2
import numpy as np

path = 'imgs/2023_3_20_12_18'

color = ColorReproduction()
color.separators = [r'\_', r'n']

color.load_capture(path, 8)
rgb_im = color.reproduccion_cie_1931(select_wavelengths=[451, 500, 525, 550, 620, 660, 740])

print(color.wavelengths)

cv2.imshow('reproducción', np.flip(rgb_im, axis=2))
cv2.waitKey(0)
