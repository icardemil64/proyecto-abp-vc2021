# Import los modulos necesarios
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Cargar las imagenes 
img0 = cv2.imread('img/pared1.jpg', 0)
img1 = cv2.imread('img/pared2.jpg', 0)
# Detectando los puntos claves, sus descriptores y calculando el matching bruto
detector = cv2.ORB_create()
kps0, fea0 = detector.detectAndCompute(img0, None)
kps1, fea1 = detector.detectAndCompute(img1, None)
matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, False)
matches = matcher.match(fea0, fea1)
# Eliminando los outlier via calculo de la Homografia y quedandose con los inlier
# realizando un calculo robusto a través de RANSAC
pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1,2)
pts1 = np.float32([kps1[m.trainIdx].pt for m in matches]).reshape(-1,2)
H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)
# Visualizando los resultados, con matching bruto y matching depurado por Homografia robusta
plt.figure()
plt.subplot(211)
plt.axis('off')
plt.title('Correspondencia Bruta')
dbg_img = cv2.drawMatches(img0, kps0, img1, kps1, matches, None)
plt.imshow(dbg_img[:,:,[2,1,0]])
plt.subplot(212)
plt.axis('off')
plt.title('Correspondencia Mejorada')
dbg_img = cv2.drawMatches(img0, kps0, img1, kps1, [m for i,m in enumerate(matches) if mask[i]], None)
plt.imshow(dbg_img[:,:,[2,1,0]])
plt.tight_layout()
plt.show()