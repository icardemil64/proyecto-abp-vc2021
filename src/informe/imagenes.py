import cv2
import numpy as np

img1 = cv2.imread("img/pared1.jpg",0)
img2 = cv2.imread("img/pared2.jpg",0)

numpy_horizontal = np.hstack((img1, img2))
numpy_horizontal_concat = np.concatenate((img1, img2), axis=1)

cv2.imshow('Escenario de pruebas', numpy_horizontal_concat)

cv2.waitKey()