from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
import argparse

import os
import cv2
import dlib
import numpy as np
from mtcnn.mtcnn import MTCNN



import face_recognition
from matlab_cp2tform import get_similarity_transform_for_cv2
from PIL import Image
from get_landmarks import get_five_points_landmarks
import net_sphere



parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--model','-m', default='../my_face_model/model/sphere20a_20171020.pth', type=str)
args = parser.parse_args()

net = getattr(net_sphere,args.net)()
net.load_state_dict(torch.load(args.model))
#net.cuda()
net.eval()
net.feature = True


import sys
from PyQt5.QtCore import *
import cv2
import numpy as np
import warnings
import DataPrepare_v1 as DataPrepare
from mtcnn.mtcnn import MTCNN
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
from mtcnn.mtcnn import MTCNN

sys.setrecursionlimit(1000000)
myFolder = os.path.split(os.path.realpath(__file__))[0]
sys.path = [os.path.join(myFolder, 'pymysql')
] + sys.path

os.chdir(myFolder)

from PyMySQL import *
#添加新人脸的线程
class AddFaceThread(QThread):
    #传出的信号为图片中人脸的位置矩形以及识别出的人名
    Bound_box = pyqtSignal(int,int,int,int)
    No_face = pyqtSignal()
    Ask_name = pyqtSignal()
    def __init__(self):
        super(AddFaceThread, self).__init__()
        self.detector = MTCNN()
        self.db = PyMySQL('localhost','root','Asd980517','WEININGFACE')
    def SetImg(self,img):
        self.img = img
        print('img ok')
        #传入图片后执行run方法
        self.start()

    def Cal_Area_Index(self,result):

        areas = []
        for face in result:
            bounding_boxes = face['box']
            areas.append(bounding_boxes[3]*bounding_boxes[2])
        return areas.index(max(areas))

    def run(self):
        print('run ok')
        print(self.img)
        result = self.detector.detect_faces(self.img)
        print(result)
        print('1')
        #如果没有检测出人脸，发出一个信号并且提前停止线程
        if len(result) == 0 :
            self.No_face.emit()
            return

        aligment_imgs = []
        temp_landmarks = []
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
        print('2')
        faces = self.alignment(faces, temp_landmarks)
        faces = np.transpose(faces, (2, 0, 1)).reshape(1, 3, 112, 96)
        faces = (faces - 127.5) / 128.0
        aligment_imgs.append(faces)
        length = len(aligment_imgs)
        aligment_imgs = np.array(aligment_imgs)
        aligment_imgs = np.reshape(aligment_imgs, (length, 3, 112, 96))
        #获取feature 向量
        output_imgs_features = self.get_imgs_features(aligment_imgs)
        print('finished feature')
        self.Bound_box.emit(bouding_boxes[1],bouding_boxes[1]+bouding_boxes[3],bouding_boxes[0],bouding_boxes[0]+bouding_boxes[2])
        print('3')

        self.db.delete_all()
        self.db.insert([name],[11],[output_imgs_features],['2018-07-07 05:23:52'])


    def cal_cosdistance(self, vec1, vec2):
        vec1 = np.reshape(vec1, (1, -1))
        vec2 = np.reshape(vec2, (-1, 1))
        length1 = np.sqrt(np.square(vec1).sum())
        length2 = np.sqrt(np.square(vec2).sum())
        cosdistance = vec1.dot(vec2) / (length1 * length2)
        cosdistance = cosdistance[0][0]
        return cosdistance

    def get_imgs_features(self, imgs_alignment):
        input_images = Variable(torch.from_numpy(imgs_alignment).float(), volatile=True)
        output_features = net(input_images)
        output_features = output_features.data.numpy()
        return output_features

    def alignment(self,src_img,src_pts):
        ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
            [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
        crop_size = (96, 112)
        src_pts = np.array(src_pts).reshape(5,2)

        s = np.array(src_pts).astype(np.float32)
        r = np.array(ref_pts).astype(np.float32)

        tfm = get_similarity_transform_for_cv2(s, r)
        face_img = cv2.warpAffine(src_img, tfm, crop_size)
        return face_img