
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import time

cap=cv2.VideoCapture(0)
detector=MTCNN()
while 1:
    t0=time.time()
    ret,img=cap.read()
    result=detector.detect_faces(img)
    if len(result)==0:
        cv2.imshow('face',img)
        continue
    for faces in result:
        bouding_boxes = faces['box']
        keypoints = faces['keypoints']
        cv2.rectangle(img, (bouding_boxes[0], bouding_boxes[1]),
                          (bouding_boxes[0] + bouding_boxes[2], bouding_boxes[1] + bouding_boxes[3]), (255, 0, 0), 2)
    img=cv2.resize(img,(900,900))
    cv2.imshow('face',img)
    t1=time.time()
    print(str(t1-t0))
    cv2.waitKey(20)