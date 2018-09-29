from __future__ import print_function
import sys
from PyQt5 import QtCore, QtGui,QtWidgets

from PyQt5.QtCore import QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtWidgets import QApplication, QLineEdit, QInputDialog, QGridLayout, QLabel, QPushButton, QFrame, QWidget,QMenu, QMainWindow
import os
sys.setrecursionlimit(1000000)


import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import time
import DataPrepare_v1 as DataPrepare
import warnings

warnings.filterwarnings('ignore')

thres = 0.5  # threshold for recognition

# load recognition model
datas = DataPrepare.ImagePrepare('images')
imgs_alignment = datas.imgs_after_alignment
imgs_features = datas.get_imgs_features(imgs_alignment)
imgs_name_list = datas.imgs_name_list
for i, img_name in enumerate(imgs_name_list):
    img_name = img_name.split('.')[0]
    imgs_name_list[i] = img_name
imgs_name_list.append('unknown')

resize_x_y = (1600, 900)
resize_face = (250, 250)
pad = 15
cap = cv2.VideoCapture(0)
frame_do = 20
frame_recog = 5
flag1 = 0
flag2 = 0


# 总窗口
def detect_face(img):
    result = datas.detector.detect_faces(img)
    for face in result:
        temp_landmarks = []
        bouding_boxes = face['box']
        keypoints = face['keypoints']
        cv2.rectangle(img, (bouding_boxes[0], bouding_boxes[1]),
                      (bouding_boxes[0] + bouding_boxes[2], bouding_boxes[1] + bouding_boxes[3]), (255, 0, 0), 2)
    return img
# detection_recognition module:
def detection_recognition(img):
    name_list=[]
    result = datas.detector.detect_faces(img)
    if len(result) == 0:
        name_list.append(-1)
        return img,name_list
    aligment_imgs = []
    originfaces = []
    # 检测，标定landmark
    for face in result:
        temp_landmarks = []
        bouding_boxes = face['box']
        keypoints = face['keypoints']

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
        tx = time.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(img, name, (result[i]['box'][0], result[i]['box'][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.putText(img, str('Age:') + str(int(face_ages)), (result[i]['box'][0] + 10, result[i]['box'][1] + 20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(img, str('time:') + str(tx), (result[i]['box'][0] + 10, result[i]['box'][1] + 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    return img,name_list

##add new face
def add_new_face(img, name):
    cv2.imwrite("./images/" + str(name) + '.jpg', img)


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        # self.face_recong = face.Recognition()
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x =0
        self.recognition_flag=False
        self.initMenu()
        self.initAnimation()

    def set_ui(self):
        self.resize(1000,800)

        #加一个窗口
        self.centralwidget = QtWidgets.QWidget()
        self.centralwidget.resize(800,600)
        self.nameLable = QLabel(" ")
        self.centralwidget.__layout_main = QtWidgets.QHBoxLayout()
        self.centralwidget.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.centralwidget.__layout_data_show = QtWidgets.QVBoxLayout()
        ###为显示图像位置添加Label
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(200, 200)
        self.label_show_camera.setFixedSize(800, 600)
        self.label_show_camera.setAutoFillBackground(False)


        self.centralwidget.__layout_main.addWidget(self.centralwidget.label_show_camera)

        self.centralwidget.label_move.raise_()

    def contextMenuEvent(self, event):
        pos = event.globalPos()
        size = self._contextMenu.sizeHint()
        x, y, w, h = pos.x(), pos.y(), size.width(), size.height()
        self._animation.stop()
        self._animation.setStartValue(QRect(x, y, 0, 0))
        self._animation.setEndValue(QRect(x, y, w, h))
        self._animation.start()
        self._contextMenu.exec_(event.globalPos())

    def initMenu(self):
        self._contextMenu = QMenu(self)
        self._contextMenu.addAction('打开相机', self.button_open_camera_click)
        self._contextMenu.addAction('识别', self.button_detection_click)
        self._contextMenu.addAction('记录', self.button_record_click)
    def initAnimation(self):
        # 按钮动画
        self._animation = QPropertyAnimation(
            self._contextMenu, b'geometry', self,
            easingCurve=QEasingCurve.Linear, duration=300)
        # easingCurve 修改该变量可以实现不同的效果


    def slot_init(self):

        self.timer_camera.timeout.connect(self.show_camera)

    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please check you have connected your camera", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        
            else:
                self.timer_camera.start(50)
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.button_open_camera.setText(u'open your camera')


    def show_camera(self):
        flag, self.image= self.cap.read()
        if self.recognition_flag==True:
            self.detect_recognition()
        show = cv2.resize(self.image, (800, 600))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        #在相机窗口上显示
        self.centralwidget.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
        # self.x += 1
        # self.label_move.move(self.x,100)

        # if self.x ==320:
        #     self.label_show_camera.raise_()
    def button_detection_click(self):
        if self.timer_camera.isActive()==False:
            msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"pleas open your camara", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)

        else:
            if self.recognition_flag==False:
                self.recognition_flag=True
                self.button_detect.setText(u'stop recognition')
            else:
                self.recognition_flag=False
                self.button_detect.setText(u'begin recognition')

    def button_record_click(self):
        if self.timer_camera.isActive()==False:
            msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"please open your camara", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            if self.recognition_flag==False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"you are not using recognition", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                file=open('record.txt','a')
                file.write('name: ')
                file.write(str(self.name_list))
                tx = time.strftime('%Y-%m-%d %H:%M:%S')
                file.write('\n')
                file.write(tx)
                file.close()

    def button_wrtieface_click(self):
        if self.timer_camera.isActive() == False:
            msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please open your camara ", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            name,ok = QInputDialog.getText(self, "Your name ", "Your name",
                                            QLineEdit.Normal, self.nameLable.text())
            if(ok and (len(name)!=0)):
                add_new_face(self.image,name)
    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"close", u"close?")

        msg.addButton(ok,QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'Yes')
        cacel.setText(u'Cancel')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()
    def detect_recognition(self):
        result = datas.detector.detect_faces(self.image)
        aligment_imgs = []
        originfaces = []
        # 检测，标定landmark
        for face in result:
            temp_landmarks = []
            bouding_boxes = face['box']
            keypoints = face['keypoints']

            cv2.rectangle(self.image, (bouding_boxes[0], bouding_boxes[1]),
                          (bouding_boxes[0] + bouding_boxes[2], bouding_boxes[1] + bouding_boxes[3]), (255, 0, 0), 2)

            faces = self.image[bouding_boxes[1]:bouding_boxes[1] + bouding_boxes[3],
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
            tx = time.strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(self.image, name, (result[i]['box'][0], result[i]['box'][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
                        1)
            cv2.putText(self.image, str('time:') + str(tx), (result[i]['box'][0] + 10, result[i]['box'][1] + 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

app = QtWidgets.QApplication(sys.argv)
ui = Ui_MainWindow()
ui.show()
ui.centralwidget.show()
sys.exit(app.exec_())