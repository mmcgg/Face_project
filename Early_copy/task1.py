import cv2
import numpy as np
import time


##参数表
Haar_front_scale=1.1            #Haar正脸图像金字塔比例，1.1~1.4
Haar_front_neibor=3             #Haar neibor参数，>=2
Haar_profile_scale=1.1
Haar_profile_neibor=3

resize_x_y=(1600,900)         #检测时如果需要resize图像的参数
resize_face=(250,250)         #检测到的人脸resize后的大小
pad=20                        #边缘填充参数

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')  # 打开分类器
profileface_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')  # 打开侧脸分类器
filename=' '#进行检测的图像文件名


##检测函数定义
def face_detector_withresize_and_write(img,face_cascade,profileface_cascade,frontscale,frontnei,profilescale,profilenei,resizeface,pad):
    t0=time.time()
    grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grey=cv2.equalizeHist(grey)         #直方图均衡化，可以去除该步骤
    grey_flip=cv2.flip(grey,1)      #翻转图像
    profileface_right=profileface_cascade.detectMultiScale(grey_flip,profilescale,profilenei)#将左脸分类器通过翻转图像的方法建立为右脸分类器
    faces=face_cascade.detectMultiScale(grey,frontscale,frontnei)
    profileface=profileface_cascade.detectMultiScale(grey,profilescale,profilenei)

    for (x,y,w,h) in faces:
        facename=str(x)+'_'+str(x+w)+'_' +str(y)+'_'+str(y+h)+'.jpg'    #文件名为人脸所在的坐标
        part=img[y-pad:y+h+pad,x-pad:x+w+pad]                   #边缘填充后的矩阵
        part=cv2.resize(part,resizeface)                        #提取人脸并且进行resize
        cv2.imwrite(facename,part)                              #写成文件的形式保存
    for (x,y,w,h) in profileface:
        profilename=str(x)+'_'+str(x+w)+'_' +str(y)+'_'+str(y+h)+'.jpg'
        part1=img[y-pad:y+h+pad,x-pad:x+w+pad]
        part1=cv2.resize(part1,resizeface)
        cv2.imwrite(profilename,part1)
    for (x,y,w,h) in profileface_right:
        draw_flip_rectangle(img,x,y,w,h,2)
        Rprofilename=str(x)+'_'+str(x+w)+'_' +str(y)+'_'+str(y+h)+'.jpg'
        size=img.shape
        width=size[1]
        part2=img[y-pad:y+h+pad,width-(x+w)-pad:width-x+pad]
        part2=cv2.resize(part2,resizeface)
        cv2.imwrite(Rprofilename,part2)
    t1=time.time()
    print('time:',str(t1-t0))


img=cv2.imread(filename)
img=cv2.resize(img,resize_x_y)

#调用函数
face_detector_withresize_and_write(img,face_cascade,profileface_cascade,Haar_front_scale,Haar_front_neibor,Haar_profile_scale,Haar_profile_neibor,resize_face,pad)