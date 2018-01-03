import numpy as np
import cv2


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        isRect = False
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
            isRect = True


            # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return isRect

im = cv2.imread('aaa.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(im, contours, -1, (0,255,0), 1)
# print hierarchy
# cv2.imshow('im', im2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
m = [1]
traversalNode = hierarchy[0][1]
#print traversalNode[0] != -1
while traversalNode[0] != -1:
    m.append(traversalNode[0])
    traversalNode = hierarchy[0][traversalNode[0]]

layer2 = []
for index in m:
    layer2.append(contours[index])

height, width = im.shape[:2]

sd = ShapeDetector()

layer3 = []
# loop over the contours
for c in layer2:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * 1)
    cY = int((M["m01"] / M["m00"]) * 1)
    shape = sd.detect(c)
    if shape:
        layer3.append(c)

cv2.drawContours(im, layer3, -1, (0,255,0), 3)

scale_im = cv2.resize(im, (width / 3, height / 3), interpolation = cv2.INTER_CUBIC)
cv2.imshow('im', scale_im)
cv2.waitKey(0)
cv2.destroyAllWindows()

