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
import numpy as np
from mtcnn.mtcnn import MTCNN
from includes.Face.matlab_cp2tform import get_similarity_transself_for_cv2
from includes.thread.AddFaceThread import AddFaceThread
from includes.thread.DetectionThread import DetectionThread
import includes.Face.net_sphere  as net_sphere
import qdarkstyle

#import network model
parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--model','-m', default='./model/sphere20a_20171020.pth', type=str)
args = parser.parse_args()


net = getattr(net_sphere,args.net)()
net.load_state_dict(torch.load(args.model))
# device = torch.device('cuda',0) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
net.to(device)
net.eval()
net.feature = True


import sys
from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import os


sys.setrecursionlimit(1000000)
myFolder = os.path.split(os.path.realpath(__file__))[0]
sys.path = [os.path.join(myFolder, 'thread')
           ,os.path.join(myFolder,'resources')
] + sys.path

os.chdir(myFolder)
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import time
import warnings
from includes.pymysql.PyMySQL import *
from Widgets.DBWidge import DBWidge
warnings.filterwarnings('ignore')

class Ui_MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)


        self.face_num  = 0
        #数据库调用
        self.dbWidge = DBWidge()
        self.dbWidge.setHidden(True)
        self.db = PyMySQL('localhost','root','Asd980517','WEININGFACE')
        #相机区域

        #人脸识别与记录线程
        self.detector = MTCNN()
        self.FaceThread = DetectionThread(self.detector,net)
        #添加新人脸的线程
        self.AddFaceThread = AddFaceThread(self.detector,net)
        #定时器
        self.timer_camera =   QTimer()
        self.timer_clear_label = QTimer()
        self.timer_dynamic_recog = QTimer()
        self.timer_long_name = QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0    #Camera used
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x =0
        self.recognition_flag=False

        #初始化
        self.initMenu()
        self.initAnimation()
        # self.setBackGround()

        self.facelabel_list = []
        self.textlabel_list = []
        self.name_list = []
        self.long_name_list = []
        self.setLabelList(self.facelabel_list,self.textlabel_list)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.timer_clear_label.start(5000)
        self.timer_long_name.start(60000)

    def set_ui(self):
        self.resize(1132,900)

        self.textBrowser = QtWidgets.QTextBrowser(self)
        self.textBrowser.setGeometry(QtCore.QRect(10, 650, 661, 151))
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.setFont(QFont("Timers",14))
        self.tabWidget = QtWidgets.QTabWidget(self)
        self.tabWidget.setGeometry(QtCore.QRect(670, 40, 371, 761))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayoutWidget = QtWidgets.QWidget(self.tab)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 0, 361, 741))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.faceLabel1 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.faceLabel1.setObjectName("faceLabel1")
        self.gridLayout.addWidget(self.faceLabel1, 0, 0, 1, 1)
        self.infoLabel3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.infoLabel3.setObjectName("infoLabel3")
        self.gridLayout.addWidget(self.infoLabel3, 2, 1, 1, 1)
        self.infoLabel1 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.infoLabel1.setObjectName("infoLabel1")
        self.gridLayout.addWidget(self.infoLabel1, 0, 1, 1, 1)
        self.faceLabel3 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.faceLabel3.setObjectName("faceLabel3")
        self.gridLayout.addWidget(self.faceLabel3, 2, 0, 1, 1)
        self.infoLabel2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.infoLabel2.setObjectName("infoLabel2")
        self.gridLayout.addWidget(self.infoLabel2, 1, 1, 1, 1)
        self.faceLabel2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.faceLabel2.setObjectName("faceLabel2")
        self.gridLayout.addWidget(self.faceLabel2, 1, 0, 1, 1)
        self.faceLabel4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.faceLabel4.setObjectName("faceLabel4")
        self.gridLayout.addWidget(self.faceLabel4, 3, 0, 1, 1)
        self.infoLabel4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.infoLabel4.setObjectName("infoLabel4")
        self.gridLayout.addWidget(self.infoLabel4, 3, 1, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        self.lcdNumber = QtWidgets.QLCDNumber(self)
        self.lcdNumber.setGeometry(QtCore.QRect(470, 40, 201, 41))
        self.lcdNumber.setObjectName("lcdNumber")
        self.lcdNumber.setDigitCount(2)
        self.camera_label = QtWidgets.QLabel(self)
        self.camera_label.setGeometry(QtCore.QRect(10, 90, 661, 551))
        self.camera_label.setObjectName("camera_labe")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 395, 81))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.pushButton_4 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout.addWidget(self.pushButton_4)
        self.pushButton_3 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setText("手动签到")
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.setText("清除列表")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("开启相机")
        self.horizontalLayout.addWidget(self.pushButton)

    def setLabelList(self,face_list,text_list):
        face_list.append(self.faceLabel1)
        face_list.append(self.faceLabel2)
        face_list.append(self.faceLabel3)
        face_list.append(self.faceLabel4)

        text_list.append(self.infoLabel1)
        text_list.append(self.infoLabel2)
        text_list.append(self.infoLabel3)
        text_list.append(self.infoLabel4)
        for text_label in text_list:
            text_label.setFont(QFont("Timer",12))



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
        self.ac_open_cama = self._contextMenu.addAction('打开相机', self.CameraOperation)
        self.ac_detection = self._contextMenu.addAction('一键签到', self.Checkin)
        self.ac_Addface = self._contextMenu.addAction('添加新人脸',self.AddFace)
        self.ac_DynamicRecog = self._contextMenu.addAction('开启动态识别',self.DynamicRecogOn)
        self.ac_dbManager = self._contextMenu.addAction('数据库操作',self.openDBmanager)
        self.ac_delete_text = self._contextMenu.addAction('删除信息显示',self.clear_all_text)
    def initAnimation(self):
        # 按钮动画
        self._animation = QPropertyAnimation(
            self._contextMenu, b'geometry', self,
            easingCurve=QEasingCurve.Linear, duration=300)
        # easingCurve 修改该变量可以实现不同的效果

    #定义信号槽
    def slot_init(self):

        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_clear_label.timeout.connect(self.del_instant_label)
        self.timer_dynamic_recog.timeout.connect(self.Checkin)
        self.timer_long_name.timeout.connect(self.del_long_name)
        #人脸识别算法完成后在右边的tab widget 中显示
        self.AddFaceThread.No_face.connect(self.TextShowNoFace)
        self.FaceThread.Bound_Name.connect(self.ShowInTab)
        self.FaceThread.Face_Count.connect(self.ShowInLCD)
        self.pushButton.clicked.connect(self.CameraOperation)
        self.pushButton_3.clicked.connect(self.Checkin)


    def ShowInLCD(self,number):
        self.lcdNumber.display(number)

    def TextShowNoFace(self):
        self.textBrowser.insertPlainText("未检测到人脸，请重试")
    def openDBmanager(self):
        if self.dbWidge.isHidden():
            self.dbWidge.setHidden(False)


    def del_instant_label(self):
        #删除第一个Label,剩余label后移动
        if not self.textlabel_list[0].text():
            return

        print(self.name_list)
        self.facelabel_list[0].clear()
        print('face_label cleared')
        name = self.textlabel_list[0].text().split('#')[1]
        self.textlabel_list[0].clear()
        print('text_label cleared')
        self.name_list.remove(name)
        print('name:',name,'removed')
        print(self.textlabel_list.__len__())
        for i in range(self.textlabel_list.__len__()-1):
            print(self.textlabel_list[i].text())
            if self.textlabel_list[i+1].text():
                print('p2')
                self.facelabel_list[i].setPixmap(self.facelabel_list[i+1].pixmap())
                self.textlabel_list[i].setText(self.textlabel_list[i+1].text())
                self.textlabel_list[i+1].clear()
                self.facelabel_list[i+1].clear()



    def setDynamicShowTime(self,t):
        self.timer_instant.start(t)


    def AddFace(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please check you have connected your camera", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            img = self.image.copy()
            self.AddFaceThread.SetImg(img)

    def DynamicRecogOn(self):
        if self.timer_camera.isActive()==False:
            msg = QtWidgets.QMessageBox.warning(self, u"warning", u"没有检测到摄像头", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            if self.timer_dynamic_recog.isActive() == False:
                self.timer_dynamic_recog.start(400)
                self.ac_DynamicRecog.setText('关闭动态识别')
            else:
                self.timer_dynamic_recog.stop()
                self.ac_DynamicRecog.setText('开启动态识别')

    def CameraOperation(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please check you have connected your camera", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        
            else:
                self.timer_camera.start(50)
                self.ac_open_cama.setText('关闭摄像头')
                self.pushButton.setText('关闭摄像头')
        else:
            if self.timer_dynamic_recog.isActive():
                self.timer_dynamic_recog.stop()
                self.ac_DynamicRecog.setText('开启动态识别')
            self.timer_camera.stop()
            self.cap.release()
            self.camera_label.clear()
            self.ac_open_cama.setText('打开摄像头')
            self.pushButton.setText('打开摄像头')
    #相机显示
    def show_camera(self):
        flag, self.image= self.cap.read()
        if self.recognition_flag==True:
            self.FaceThread.SetImg(self.image,method=1)

        show = cv2.resize(self.image, (800, 600))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],QImage.Format_RGB888 )
        self.camera_label.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def Checkin(self):
        if self.timer_camera.isActive()==False:
            msg = QtWidgets.QMessageBox.warning(self, u"warning", u"没有检测到摄像头", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)

        else:
            #启动识别算法线程
            self.RecogImage = self.image.copy()
            self.FaceThread.SetImg(self.image)

    # def button_wrtieface_click(self):
    #     if self.timer_camera.isActive() == False:
    #         msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please open your camara ", buttons=QtWidgets.QMessageBox.Ok,
    #                                             defaultButton=QtWidgets.QMessageBox.Ok)
    #     else:
    #         name,ok = QInputDialog.getText(self, "Your name ", "Your name",
    #                                         QLineEdit.Normal, self.nameLable.text())
    #         if(ok and (len(name)!=0)):
    #             add_new_face(self.image,name)
    def ShowInTab(self,bound0,bound1,bound2,bound3,name):

        face = self.RecogImage[bound1:bound1 + bound3,
                    bound0:bound0 + bound2]
        show = cv2.resize(face, (200,200))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)

        pix = QPixmap.fromImage(showImage)
        print(self.name_list)
        if self.check_name(name)==True:
            for i,text_label in enumerate(self.textlabel_list):
                if not text_label.text():
                    self.facelabel_list[i].setPixmap(pix)
                    tx = time.strftime('%Y-%m-%d\n%H:%M:%S')
                    tx1 = time.strftime('%Y-%m-%d %H:%M:%S')
                    all_str = '姓名:#' + name + '#\n' + '时间:' + tx
                    text_label.setText(all_str)
                    break

    def check_name(self,name):
        if name not in self.long_name_list:
            if name == 'Unknown':
                str_1 = '检测到未知人员于' + tx1 + '出现\n'
                self.textBrowser.insertPlainText(str_1)
            else:
                self.long_name_list.append(name)
                tx1 = time.strftime('%Y-%m-%d %H:%M:%S')
                str_1 = '检测到#' + name + '#于' + tx1 + '出现\n'
                self.textBrowser.insertPlainText(str_1)
        if name not in self.name_list:
            self.name_list.append(name)
            return True

        else:
            return False

    def del_long_name(self):
        self.long_name_list.clear()

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"关闭?")

        msg.addButton(ok,QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'是')
        cacel.setText(u'否')

        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()

    def clear_all_label(self,face_list,text_list):
        for label in face_list:
            label.clear()

        for label in text_list:
            text_list.clear()

    def clear_all_text(self):
        self.textBrowser.clear()

app = QtWidgets.QApplication(sys.argv)
ui = Ui_MainWindow()
ui.show()
sys.exit(app.exec_())
