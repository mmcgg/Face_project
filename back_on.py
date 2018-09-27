


from __future__ import print_function
import sys
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import time
import DataPrepare_v1 as DataPrepare
import warnings
warnings.filterwarnings('ignore')



thres=0.5              #threshold for recognition

#load recognition model
datas=DataPrepare.ImagePrepare('images')
imgs_alignment=datas.imgs_after_alignment
imgs_features=datas.get_imgs_features(imgs_alignment)

imgs_name_list=datas.imgs_name_list
for i, img_name in enumerate(imgs_name_list):
    img_name = img_name.split('.')[0]
    imgs_name_list[i] = img_name
imgs_name_list.append('unknown')

resize_x_y=(1600,900)      
resize_face=(250,250)         
pad=15                       
cap=cv2.VideoCapture(0)       
frame_do=20               
frame_recog=5            
flag1=0                     
flag2=0                     

#总窗口

#detection_recognition module:
def detection_recognition(img):
    result = datas.detector.detect_faces(img)
    if len(result) == 0:
        return img,[]
    aligment_imgs = []
    originfaces= []
    #检测，标定landmark
    for face in result:
        temp_landmarks = []
        bouding_boxes = face['box']
        keypoints = face['keypoints']
        if bouding_boxes[1]-pad<=0:
            bouding_boxes[1]=1
        else :
            bouding_boxes[1]=bouding_boxes[1]-pad
        if bouding_boxes[0]-pad<=0:
            bouding_boxes[0]=1
        else :
            bouding_boxes[0]=bouding_boxes[0]-pad
        if (bouding_boxes[1]+bouding_boxes[3]+pad)<img.shape[0]:
            bouding_boxes[3]=bouding_boxes[3]+pad*2
        if (bouding_boxes[0]+bouding_boxes[2]+pad )<img.shape[1]:
            bouding_boxes[2]=bouding_boxes[2]+pad*2

        cv2.rectangle(img, (bouding_boxes[0], bouding_boxes[1]),
                      (bouding_boxes[0] + bouding_boxes[2], bouding_boxes[1] + bouding_boxes[3]), (255, 0, 0), 2)
 
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
    output_imgs_features = datas.get_imgs_features(aligment_imgs)
    cos_distances_list = []
    result_index = []
    name_list=[]
    for img_feature in output_imgs_features:
        cos_distance_list = [datas.cal_cosdistance(img_feature, test_img_feature) for test_img_feature in imgs_features]
        cos_distances_list.append(cos_distance_list)
    for imgfeature in cos_distances_list:
        if max(imgfeature) < thres:
            result_index.append(-1)
        else:
            result_index.append(imgfeature.index(max(imgfeature)))
    for i, index in enumerate(result_index):
        name = imgs_name_list[i]
        name_list.append(name)
        face_ages = age.predict(originfaces[i])
        tx=time.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(img, name, (result[i]['box'][0], result[i]['box'][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(img, str('time:') + str(tx), (result[i]['box'][0] + 10, result[i]['box'][1] + 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    return img,name_list


##add new face
def add_new_face(img,name):
    cv2.imwrite("./images/"+str(name)+'.jpg',img)




