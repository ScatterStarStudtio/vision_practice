import numpy as np
import cv2

class ShapeDetector:
    def __init__(self):
        pass

    def isRect_Area_Ge(self, c, area):
        isRect = False
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 4 and cv2.contourArea(approx) >= area and cv2.isContourConvex(approx):
            isRect = True

        return isRect

im = cv2.imread('aaa.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
m = [1]
traversalNode = hierarchy[0][1]
while traversalNode[0] != -1:
    m.append(traversalNode[0])
    traversalNode = hierarchy[0][traversalNode[0]]

layer2 = []
for index in m:
    layer2.append(contours[index])

sd = ShapeDetector()

layer3 = []
# loop over the contours
for c in layer2:
    if sd.isRect_Area_Ge(c, 3000):
        layer3.append(c)

print(len(layer3))

cv2.drawContours(im, layer3, -1, (0,255,0), 3)

height, width = im.shape[:2]

scale_im = cv2.resize(im, (int(width / 3), int(height / 3)), interpolation = cv2.INTER_CUBIC)
cv2.imshow('im', scale_im)
cv2.waitKey(0)
cv2.destroyAllWindows()

