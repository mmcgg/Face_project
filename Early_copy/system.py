from __future__ import print_function
import cv2
import numpy as np
import face_recognition
import dlib
import time
import Face
import DataPrepare
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
import argparse
from matlab_cp2tform import get_similarity_transform_for_cv2
from get_landmarks import get_five_points_landmarks
import net_sphere
import mtcnn
##前端系统搭建
##参数表
thres=0.53                    #识别阈值
datas=DataPrepare.ImagePrepare('images')
imgs_alignment=datas.imgs_after_alignment
imgs_features=datas.get_imgs_features(imgs_alignment)

imgs_name_list=datas.imgs_name_list
for i, img_name in enumerate(imgs_name_list):
    img_name = img_name.split('.')[0]
    imgs_name_list[i] = img_name
imgs_name_list.append('unknown')
Haar_front_scale=1.1            #Haar正脸图像金字塔比例，1.1~1.4
Haar_front_neibor=8             #Haar neibor参数，>=2
Haar_profile_scale=1.1
Haar_profile_neibor=3

resize_x_y=(1600,900)         #检测时如果需要resize图像的参数
resize_face=(250,250)         #检测到的人脸resize后的大小
pad=40                        #边缘填充参数
flag_max=10                   #帧数
face_cascade = cv2.CascadeClassifier('face.xml')  # 打开分类器
eye_cascade=cv2.CascadeClassifier('xml/haarcascade_eye.xml')#眼部检测
nose_cascade=cv2.CascadeClassifier('xml/haarcascade_mcs_nose.xml')#鼻子检测
mouth_cascade=cv2.CascadeClassifier('xml/haarcascade_mcs_mouth.xml')#嘴部检测
#landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat.bz2')       #dlib的人脸landmark
cap = cv2.VideoCapture(0)
flag=0
while 1:
    t0=time.time()
    ret,img=cap.read()                          #读取帧

    aligment_imgs=[]
    detected_faces,faces_axis=Face.face_detector_with_img_out_frontal(img,face_cascade,Haar_front_scale,Haar_front_neibor,resize_face,pad)

    print(len(detected_faces))
    landmarks=[]
    if (len(detected_faces)==0):
        cv2.imshow('face',img)
        cv2.waitKey(10)
        continue
    for faces in detected_faces:

        landmark_list=face_recognition.face_landmarks(faces)
        landlist=[]
        print('landmark'+str(len(landmark_list)))
        if len(landmark_list)==0:
            cv2.imshow('face',img)
            cv2.waitKey(10)
            continue
        landmark=landmark_list[0]
        firstp=landmark['left_eye'][1]
        landlist.append(firstp[0])
        landlist.append(firstp[1])
        secp=landmark['right_eye'][1]
        landlist.append(secp[0])
        landlist.append(secp[1])
        third=landmark['nose_bridge'][3]
        landlist.append(third[0])
        landlist.append(third[1])
        forth=landmark['top_lip'][0]
        landlist.append(forth[0])
        landlist.append(forth[1])
        fifth=landmark['top_lip'][6]
        landlist.append(fifth[0])
        landlist.append(fifth[1])
        landmarks.append(landlist)
        #print(landlist)
        #landlist=[97,108,140,105,114,134,108,154,145,153]
        faces=DataPrepare.alignment(faces,landlist)
        faces=np.transpose(faces,(2,0,1)).reshape(1,3,112,96)
        faces=(faces-127.5)/128.0
        aligment_imgs.append(faces)
        #eyes=eye_cascade.detectMultiScale(faces,1.3,18)
        #print(str(eyes))
        #for (ex,ey,ew,eh) in eyes:
        #    cv2.rectangle(faces,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
        #nose=nose_cascade.detectMultiScale(faces,1.3,18)
        #print(str(nose))
        #for (ex, ey, ew, eh) in nose:
        #    cv2.rectangle(faces, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
        #mouth=mouth_cascade.detectMultiScale(faces,1.3,18)
        #print(str(mouth))
        #for (ex, ey, ew, eh) in mouth:
        #    cv2.rectangle(faces, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
        #cv2.imshow('face',faces)
        #cv2.waitKey(20)
    length = len(aligment_imgs)
    if length!=len(detected_faces):
        cv2.imshow('face',img)
        cv2.waitKey(10)
        continue
    print('alignfaces:'+str(length))
    aligment_imgs=np.array(aligment_imgs)#转换为np矩阵
    if aligment_imgs.shape==(0,):
        cv2.imshow('face',img)
        cv2.waitKey(10)
        continue
    aligment_imgs=np.reshape(aligment_imgs,(length,3,112,96))
    output_imgs_features = datas.get_imgs_features(aligment_imgs)
    cos_distances_list = []
    result_index=[]

    for img_feature in output_imgs_features:
        cos_distance_list = [datas.cal_cosdistance(img_feature, test_img_feature) for test_img_feature in imgs_features]
        cos_distances_list.append(cos_distance_list)
    for imgfeature in cos_distances_list:
        if max(imgfeature)<thres:
            result_index.append(-1)
        else:
            result_index.append(imgfeature.index(max(imgfeature)))
    print(result_index)
    print(cos_distances_list)
    for i,index in enumerate(result_index):
        name=imgs_name_list[index]
        cv2.putText(img,name,(faces_axis[i][0],faces_axis[i][1]),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)

    cv2.imshow('face', img)
    t1=time.time()
    print('time:'+str(t1-t0))
    detected_faces=[]                               #清空
    landmarks=[]
    cv2.waitKey(50) & 0xFF==ord('q')




