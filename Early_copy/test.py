import cv2
import numpy as np

test_img=cv2.imread('test1.jpg')
test_img=cv2.resize(test_img,(200,200))
print(test_img)
img=np.zeros((800,800,3))
img[0:test_img.shape[0],0:test_img.shape[1]]=test_img
hebing=np.hstack([test_img,test_img])
cv2.imshow('test',img)
cv2.imshow('teststack',hebing)
cv2.waitKey(0)
cv2.destroyAllWindows()