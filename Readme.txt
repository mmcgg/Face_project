#前端(程序入口）： version2_3_4.py (python3.6)
#后端（人脸识别）: back_on.py
#主要依赖包：
  pyqt5
  torch
  dlib
  face_recognition
  mtcnn
  opencv-python
  tensorflow
  matplotlib

说明：
#人脸库: 
images文件夹中 .jpg格式

实现功能:
人脸识别
年龄预测
添加新人脸
记录姓名与时间 （record.txt）


#目前存在的bug/不足：
1.界面需要美化（前端）
2.添加新名字的时候，需要保证检测到人脸，添加人脸后需要重启程序
3.如果prepare data报错，说明images中有图片未检测到人脸，需要删除后重新录入（
建议打开人脸识别的同时进行人脸添加）


made by 郭远帆 罗翔中 丁智勇
