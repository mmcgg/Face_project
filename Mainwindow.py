from __future__ import print_function
import sys
from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtWidgets import QApplication, QLineEdit, QInputDialog, QGridLayout, QLabel, QPushButton, QFrame, QWidget,QMenu
from PyQt5.QtCore import  QThread, QThreadPool
import os

sys.setrecursionlimit(1000000)


import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import time
import DataPrepare_v1 as DataPrepare
import warnings

warnings.filterwarnings('ignore')



resize_x_y = (1600, 900)
resize_face = (250, 250)


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

##add new face
def add_new_face(img, name):
    cv2.imwrite("./images/" + str(name) + '.jpg', img)


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        #相机区域
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.resize(1022, 670)

        self.set_ui()
        self.slot_init()

        self.__flag_work = 0
        self.x =0
        self.recognition_flag=False

        #初始化右键下拉菜单
        self.initMenu()
        self.initAnimation()
    def set_ui(self):
        self.nameLable = QLabel(" ")
        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()
        #tab菜单加载
        self.showface = QtWidgets.QTabWidget(self)
        self.showface.setGeometry(QtCore.QRect(740, 80, 351, 691))
        self.showface.setObjectName("showface")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(20, 20, 301, 611))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.listView = QtWidgets.QListView(self.horizontalLayoutWidget_2)
        self.listView.setObjectName("listView")
        self.horizontalLayout_2.addWidget(self.listView)
        self.showface.addTab(self.tab, "第一页")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.listView_2 = QtWidgets.QListView(self.tab_2)
        self.listView_2.setGeometry(QtCore.QRect(20, 20, 299, 609))
        self.listView_2.setObjectName("listView_2")
        self.showface.addTab(self.tab_2, "第二页")

        #用于显示图像的Label
        self.label_show_camera = QtWidgets.QLabel()

        
        self.label_show_camera.resize(800,600)
        self.label_show_camera.setAutoFillBackground(False)


        self.__layout_main.addWidget(self.label_show_camera)

        self.__layout_main.addWidget(self.showface)
        self.setLayout(self.__layout_main)
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
        self.ac_open_cama = self._contextMenu.addAction('打开相机', self.CameraOperation())
        self.ac_detection = self._contextMenu.addAction('识别', self.RecognitionOn())
        self.ac_record = self._contextMenu.addAction('记录', self.Record())
    def initAnimation(self):
        # 按钮动画
        self._animation = QPropertyAnimation(
            self._contextMenu, b'geometry', self,
            easingCurve=QEasingCurve.Linear, duration=300)
        # easingCurve 修改该变量可以实现不同的效果


    def slot_init(self):

        self.timer_camera.timeout.connect(self.show_camera)

    #打开相机操作
    def CameraOperation(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please check you have connected your camera", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        
            else:
                self.timer_camera.start(50)
                self.ac_open_cama.setText('关闭相机')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.ac_open_cama.setText('打开相机')
    def show_camera(self):
        flag, self.image= self.cap.read()
        if self.recognition_flag ==True:
            self.image = self.detect_recognition(self.image)
        show = cv2.resize(self.image, (800, 600))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def RecognitionOn(self):
        if self.timer_camera.isActive()==False:
            msg = QtWidgets.QMessageBox.warning(self, u"warning", u"没有检测到摄像头", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)

        else:
            if self.recognition_flag==False:
                self.recognition_flag=True

            else:
                self.recognition_flag=False


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

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"关闭?")

        msg.addButton(ok,QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'是')
        cacel.setText(u'否')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()


class DetectionThread(QThread):
    #传出的信号为图片中人脸的位置矩形以及识别出的人名
    self.Bound_Name = pyqtSignal[int, int, int, int,str]
    def __init__(self):
        super(DetectionThread, self).__init__()
        #为自己导入模型
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

    def SetImg(self,img):
        self.img = img
        #传入图片后执行run方法
        self.start()
    def run(self):
        result = datas.detector.detect_faces(self.img)
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

    def finished(self):
        pass


app = QtWidgets.QApplication(sys.argv)
ui = Ui_MainWindow()
ui.show()
sys.exit(app.exec_())