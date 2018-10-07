import sys
from PyQt5.QtCore import *
import cv2
import numpy as np
import warnings
import DataPrepare_v1 as DataPrepare
from mtcnn.mtcnn import MTCNN


#识别算法的线程
class DetectionThread(QThread):
    #传出的信号为图片中人脸的位置矩形以及识别出的人名
    Bound_Name = pyqtSignal(int,int,int,int,str)
    No_face = pyqtSignal()
    def __init__(self):
        super(DetectionThread, self).__init__()
        #为自己导入模型

    def SetImg(self,img):
        self.img = img
        #传入图片后执行run方法
        self.start()
    def run(self):
        result = self.datas.detector.detect_faces(self.img)
        #如果没有检测出人脸，发出一个信号并且提前停止线程
        if len(result) == 0 :
            self.No_face.emit()
            return
        aligment_imgs = []
        originfaces = []
        # 检测，标定landmark
        for face in result:
            temp_landmarks = []
            bouding_boxes = face['box']
            keypoints = face['keypoints']

            faces = img[bouding_boxes[1]:bouding_boxes[1] + bouding_boxes[3],
                    bouding_boxes[0]:bouding_boxes[0] + bouding_boxes[2]]
            originfaces.append(faces)
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
        cos_distances_list = []
        result_index = []
        for img_feature in output_imgs_features:
            cos_distance_list = [datas.cal_cosdistance(img_feature, test_img_feature) for test_img_feature in
                                 imgs_features]
            cos_distances_list.append(cos_distance_list)
        for imgfeature in cos_distances_list:
            if max(imgfeature) < thres:
                result_index.append(-1)
            else:
                result_index.append(imgfeature.index(max(imgfeature)))
        for i, index in enumerate(result_index):
            name = imgs_name_list[i]
            bound = result[i]['box']
            #发送信号
            self.Bound_Name.emit(bound[1],bound[1]+bound[3],bound[0],bound[0]+bound[2],name)

