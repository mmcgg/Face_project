import sys
from PyQt5.QtCore import *
import cv2
import numpy as np
import warnings
import DataPrepare_v1 as DataPrepare
from mtcnn.mtcnn import MTCNN


#添加新人脸的线程
class AddFaceThread(QThread):
    #传出的信号为图片中人脸的位置矩形以及识别出的人名
    Bound_box = pyqtSignal(int,int,int,int)
    No_face = pyqtSignal()
    Ask_name = pyqtSignal()
    def __init__(self):
        super(AddFaceThread, self).__init__()
        #为自己导入模型
        self.thres = 0.5  # threshold for recognition

        # load recognition model
        self.datas = DataPrepare.ImagePrepare('images')
        self.imgs_alignment = self.datas.imgs_after_alignment
        self.imgs_features = self.datas.get_imgs_features(self.imgs_alignment)
        self.imgs_name_list = self.datas.imgs_name_list
        for i, img_name in enumerate(self.imgs_name_list):
            self.img_name = img_name.split('.')[0]
            self.imgs_name_list[i] = self.img_name
        self.imgs_name_list.append('unknown')
    def Refresh(self,thres = 0.5):
        #为自己导入模型
        self.thres = thres  # threshold for recognition

        # load recognition model
        self.datas = DataPrepare.ImagePrepare('images')
        self.imgs_alignment = self.datas.imgs_after_alignment
        self.imgs_features = self.datas.get_imgs_features(imgs_alignment)
        self.imgs_name_list = self.datas.imgs_name_list
        for i, img_name in enumerate(self.imgs_name_list):
            self.img_name = img_name.split('.')[0]
            self.imgs_name_list[i] = self.img_name
        self.imgs_name_list.append('unknown')
    def SetImg(self,img):
        self.img = img
        #传入图片后执行run方法
        self.start()

    def Cal_Area_Index(self,result):

        areas = []
        for face in result:
            bounding_boxes = face['box']
            areas.append(bounding_boxes[3]*bounding_boxes[2])
        return areas.index(max(areas))

    def getInfo(self,name):
        self.name = name
    def run(self):

        result = self.datas.detector.detect_faces(self.img)
        #如果没有检测出人脸，发出一个信号并且提前停止线程
        if len(result) == 0 :
            self.No_face.emit()
            return
        aligment_imgs = []

        maxIndex = self.Cal_Area_Index(result)
        face = result[maxIndex]

        bouding_boxes = face['box']
        keypoints = face['keypoints']

        faces = self.img[bouding_boxes[1]:bouding_boxes[1] + bouding_boxes[3],
                    bouding_boxes[0]:bouding_boxes[0] + bouding_boxes[2]]

        lefteye = keypoints['left_eye']
        righteye = keypoints['right_eye']
        nose = keypoints['nose']
        mouthleft = keypoints['mouth_left']
        mouthright = keypoints['mouth_right']
        temp_landmarks.append(lefteye[0])
        temp_landmarks.append(lefteye[1])
        temp_landmarks.append(righteye[0])
        temp_landmarks.append(righteye[1])
        temp_landmarks.append(nose[0])
        temp_landmarks.append(nose[1])
        temp_landmarks.append(mouthleft[0])
        temp_landmarks.append(mouthleft[1])
        temp_landmarks.append(mouthright[0])
        temp_landmarks.append(mouthright[1])

        for i, num in enumerate(temp_landmarks):
            if i % 2:
                temp_landmarks[i] = num - bouding_boxes[1]
            else:
                temp_landmarks[i] = num - bouding_boxes[0]

        faces = DataPrepare.alignment(faces, temp_landmarks)
        faces = np.transpose(faces, (2, 0, 1)).reshape(1, 3, 112, 96)
        faces = (faces - 127.5) / 128.0
        aligment_imgs.append(faces)
        length = len(aligment_imgs)
        aligment_imgs = np.array(aligment_imgs)
        aligment_imgs = np.reshape(aligment_imgs, (length, 3, 112, 96))
        output_imgs_features = self.datas.get_imgs_features(aligment_imgs)
        self.Bound_box.emit(bouding_boxes[1],bouding_boxes[1]+bouding_boxes[3],bouding_boxes[0],bouding_boxes[0]+bouding_boxes[2])





