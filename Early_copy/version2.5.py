#version 2.5
#独立识别模块
from __future__ import print_function
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
import DataPrepare_v1 as DataPrepare


thres=0.5              #threshold for recognition
datas=DataPrepare.ImagePrepare('images')
imgs_alignment=datas.imgs_after_alignment
imgs_features=datas.get_imgs_features(imgs_alignment)

imgs_name_list=datas.imgs_name_list
for i, img_name in enumerate(imgs_name_list):
    img_name = img_name.split('.')[0]
    imgs_name_list[i] = img_name
imgs_name_list.append('unknown')

resize_x_y=(1600,900)         #检测时如果需要resize图像的参数
resize_face=(250,250)         #检测到的人脸resize后的大小
pad=15                        #边缘填充参数
cap=cv2.VideoCapture(0)       #打开内置摄像头
frame_do=20                 #检测人脸的帧数
frame_recog=5              #识别人脸的帧数
flag1=0                     #检测flag
flag2=0                     #识别flag

#总窗口

#detection/recognition模块
def detection_recognition(img):
    result = datas.detector.detect_faces(img)
    if len(result) == 0:
        return img
    aligment_imgs = []

    #检测，标定landmark
    for face in result:
        temp_landmarks = []
        bouding_boxes = face['box']
        keypoints = face['keypoints']
        cv2.rectangle(img, (bouding_boxes[0], bouding_boxes[1]),
                      (bouding_boxes[0] + bouding_boxes[2], bouding_boxes[1] + bouding_boxes[3]), (255, 0, 0), 2)
        #检测到的人脸在img中的坐标
        faces = img[bouding_boxes[1]:bouding_boxes[1] + bouding_boxes[3],
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
    aligment_imgs = np.array(aligment_imgs)  # 转换为np矩阵
    aligment_imgs = np.reshape(aligment_imgs, (length, 3, 112, 96))
    output_imgs_features = datas.get_imgs_features(aligment_imgs)
    cos_distances_list = []
    result_index = []

    for img_feature in output_imgs_features:
        cos_distance_list = [datas.cal_cosdistance(img_feature, test_img_feature) for test_img_feature in imgs_features]
        cos_distances_list.append(cos_distance_list)
    for imgfeature in cos_distances_list:
        if max(imgfeature) < thres:
            result_index.append(-1)
        else:
            result_index.append(imgfeature.index(max(imgfeature)))
    for i, index in enumerate(result_index):
        name = imgs_name_list[index]
        cv2.putText(img, name, (result[i]['box'][0], result[i]['box'][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    return img

#重构detection/recognition模块，独立出来
def detection_mtcnn(img):
    result = datas.detector.detect_faces(img)
    if len(result) == 0:
        return 0,0,0
    aligment_imgs = []
    origin_faces=[]
    # 检测，标定landmark
    for face in result:
        temp_landmarks = []
        bouding_boxes = face['box']
        keypoints = face['keypoints']

        '''
        cv2.rectangle(img, (bouding_boxes[0], bouding_boxes[1]),
                      (bouding_boxes[0] + bouding_boxes[2], bouding_boxes[1] + bouding_boxes[3]), (255, 0, 0), 2)
        '''
        # 检测到的人脸在img中的坐标
        faces = img[int(bouding_boxes[1]*0.8):int((bouding_boxes[1] + bouding_boxes[3])*1.1),
                int(bouding_boxes[0]*0.8): int ((bouding_boxes[0] + bouding_boxes[2])*1.1) ]
        origin_faces.append(faces)
        faces = img[bouding_boxes[1]:bouding_boxes[1] + bouding_boxes[3],
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

    return 1,aligment_imgs,origin_faces

#重构识别模块，独立出来
def recognition(aligment_imgs):
    length = len(aligment_imgs)
    aligment_imgs = np.array(aligment_imgs)  # 转换为np矩阵
    aligment_imgs = np.reshape(aligment_imgs, (length, 3, 112, 96))
    output_imgs_features = datas.get_imgs_features(aligment_imgs)
    cos_distances_list = []
    result_index = []

    for img_feature in output_imgs_features:
        cos_distance_list = [datas.cal_cosdistance(img_feature, test_img_feature) for test_img_feature in imgs_features]
        cos_distances_list.append(cos_distance_list)
    for imgfeature in cos_distances_list:
        if max(imgfeature) < thres:
            result_index.append(-1)
        else:
            result_index.append(imgfeature.index(max(imgfeature)))

    return result_index             #返回值为人脸编号

def put_text(img,index):
    name = imgs_name_list[index]
    cv2.putText(img, name, (30,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    return img



ret,img=cap.read()
img_after_detection_recognition=detection_recognition(img)


while 1:
    ret,img=cap.read()
    all_win=np.hstack([img,img_after_detection_recognition])

    if flag1<frame_do:
        flag1=flag1+1
        cv2.imshow('window',all_win)
        cv2.waitKey(20)
        continue
    img_after_detection_recognition=detection_recognition(img)
    flag1=0


    ret,aligment_imgs,origin_faces=detection_mtcnn(img)
    if ret!=0 :
        facename = 0
        result_index=recognition(aligment_imgs)
        for index,item in enumerate (origin_faces):
            origin_faces[index]=put_text(origin_faces[index],result_index[index])
            cv2.imshow(str(facename),item)

        cv2.waitKey(20)

