from typing import List
import cv2
import numpy as np
from matplotlib import pyplot as plt

def drawlines(img1,img2,lines,pts1,pts2):
  ''' img1 - image on which we draw the epilines for the points in img2
      lines - corresponding epilines '''
  r,c = img1.shape
  print(r)
  img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
  img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
  for r,pt1,pt2 in zip(lines,pts1,pts2):
    color = tuple(np.random.randint(0,255,3).tolist())
    x0,y0 = map(int, [0, -r[2]/r[1] ])
    x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
    #img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
    img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
    img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
  return img1,img2


img1 = cv2.imread("img/pared1.jpg",0)
img2 = cv2.imread("img/pared2.jpg",0)

#ORB Detector
orb = cv2.ORB_create()
kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

#Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(des1,des2)
pts1 = []
pts2 = []

for m in matches:
    pts1.append(kp1[m.queryIdx].pt)
    pts2.append(kp2[m.trainIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

list_validos = []
for m in matches:
  if m.queryIdx in pts1:
    list_validos.append(m)

list_validos = sorted(list_validos,key = lambda x:x.distance)
""""
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()
"""

result = cv2.drawMatches(img1, kp1, img2, kp2, list_validos[:30], None, flags=2)

cv2.imshow("Detector ORB",result)
cv2.waitKey(0)
cv2.destroyAllWindows()