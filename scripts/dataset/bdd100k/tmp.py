import cv2

img = cv2.imread('E:/dl/datasets/bdd100k/images/1d33c83b-71e1ea1c.jpg')

cv2.rectangle(img, (0, 2), (1279, 719), (0, 0, 255), 2)
cv2.imshow('1', img)
cv2.waitKey(0)