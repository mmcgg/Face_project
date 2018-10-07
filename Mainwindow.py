from __future__ import print_function
import sys
from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import os


sys.setrecursionlimit(1000000)
myFolder = os.path.split(os.path.realpath(__file__))[0]
sys.path = [os.path.join(myFolder, 'pymysql')
] + sys.path

os.chdir(myFolder)
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import time
import DataPrepare_v1 as DataPrepare
import warnings
from PyMySQL import *

from DetectionThread import DetectionThread
from AddFaceThread import AddFaceThread
warnings.filterwarnings('ignore')

class Ui_MainWindow(QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)



        self.db = PyMySQL('localhost','root','Asd980517','WEININGFACE')
        #相机区域
        #人脸识别算法线程
        self.FaceThread = DetectionThread()
        #添加新人脸的线程
        self.AddFaceThread = AddFaceThread()
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
        self.resize(1114, 861)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 230, 731, 561))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.MainCameraLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.MainCameraLayout.setContentsMargins(0, 0, 0, 0)
        self.MainCameraLayout.setObjectName("MainCameraLayout")
        self.MainCameraLabel = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.MainCameraLabel.setObjectName("MainCameraLabel")
        self.MainCameraLayout.addWidget(self.MainCameraLabel)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(730, 10, 381, 851))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.TabLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.TabLayout.setContentsMargins(0, 0, 0, 0)
        self.TabLayout.setObjectName("TabLayout")
        self.FaceTab = QtWidgets.QTabWidget(self.horizontalLayoutWidget_2)
        self.FaceTab.setObjectName("FaceTab")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(0, 0, 191, 191))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.FaceLabel1_1 = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.FaceLabel1_1.setObjectName("FaceLabel1_1")
        self.horizontalLayout_3.addWidget(self.FaceLabel1_1)
        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(0, 200, 191, 191))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.FaceLabel1_2 = QtWidgets.QLabel(self.horizontalLayoutWidget_4)
        self.FaceLabel1_2.setObjectName("FaceLabel1_2")
        self.horizontalLayout_4.addWidget(self.FaceLabel1_2)
        self.horizontalLayoutWidget_5 = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(0, 400, 191, 191))
        self.horizontalLayoutWidget_5.setObjectName("horizontalLayoutWidget_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_5)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.FaceLabel1_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_5)
        self.FaceLabel1_3.setObjectName("FaceLabel1_3")
        self.horizontalLayout_5.addWidget(self.FaceLabel1_3)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(200, 0, 141, 191))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.TextLabel1_1 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.TextLabel1_1.setObjectName("TextLabel1_1")
        self.verticalLayout_2.addWidget(self.TextLabel1_1)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(200, 200, 141, 191))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.TextLabel1_2 = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        self.TextLabel1_2.setObjectName("TextLabel1_2")
        self.verticalLayout_3.addWidget(self.TextLabel1_2)
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(200, 400, 141, 191))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.TextLabel1_3 = QtWidgets.QLabel(self.verticalLayoutWidget_4)
        self.TextLabel1_3.setObjectName("TextLabel1_3")
        self.verticalLayout_4.addWidget(self.TextLabel1_3)
        self.horizontalLayoutWidget_6 = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget_6.setGeometry(QtCore.QRect(0, 600, 191, 191))
        self.horizontalLayoutWidget_6.setObjectName("horizontalLayoutWidget_6")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_6)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.FaceLabel1_4 = QtWidgets.QLabel(self.horizontalLayoutWidget_6)
        self.FaceLabel1_4.setObjectName("FaceLabel1_4")
        self.horizontalLayout_6.addWidget(self.FaceLabel1_4)
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(200, 600, 141, 191))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.TextLabel1_4 = QtWidgets.QLabel(self.verticalLayoutWidget_5)
        self.TextLabel1_4.setObjectName("TextLabel1_4")
        self.verticalLayout_8.addWidget(self.TextLabel1_4)
        self.FaceTab.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayoutWidget_6 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_6.setGeometry(QtCore.QRect(200, 200, 141, 191))
        self.verticalLayoutWidget_6.setObjectName("verticalLayoutWidget_6")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_6)
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.TextLabel2_2 = QtWidgets.QLabel(self.verticalLayoutWidget_6)
        self.TextLabel2_2.setObjectName("TextLabel2_2")
        self.verticalLayout_9.addWidget(self.TextLabel2_2)
        self.horizontalLayoutWidget_7 = QtWidgets.QWidget(self.tab_2)
        self.horizontalLayoutWidget_7.setGeometry(QtCore.QRect(0, 400, 191, 191))
        self.horizontalLayoutWidget_7.setObjectName("horizontalLayoutWidget_7")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_7)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.FaceLabel2_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        self.FaceLabel2_3.setObjectName("FaceLabel2_3")
        self.horizontalLayout_7.addWidget(self.FaceLabel2_3)
        self.verticalLayoutWidget_7 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_7.setGeometry(QtCore.QRect(200, 0, 141, 191))
        self.verticalLayoutWidget_7.setObjectName("verticalLayoutWidget_7")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_7)
        self.verticalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.TextLabel2_1 = QtWidgets.QLabel(self.verticalLayoutWidget_7)
        self.TextLabel2_1.setObjectName("TextLabel2_1")
        self.verticalLayout_10.addWidget(self.TextLabel2_1)
        self.horizontalLayoutWidget_8 = QtWidgets.QWidget(self.tab_2)
        self.horizontalLayoutWidget_8.setGeometry(QtCore.QRect(0, 200, 191, 191))
        self.horizontalLayoutWidget_8.setObjectName("horizontalLayoutWidget_8")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_8)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.FaceLabel2_2 = QtWidgets.QLabel(self.horizontalLayoutWidget_8)
        self.FaceLabel2_2.setObjectName("FaceLabel2_2")
        self.horizontalLayout_8.addWidget(self.FaceLabel2_2)
        self.horizontalLayoutWidget_9 = QtWidgets.QWidget(self.tab_2)
        self.horizontalLayoutWidget_9.setGeometry(QtCore.QRect(0, 0, 191, 191))
        self.horizontalLayoutWidget_9.setObjectName("horizontalLayoutWidget_9")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_9)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.FaceLabel2_1 = QtWidgets.QLabel(self.horizontalLayoutWidget_9)
        self.FaceLabel2_1.setObjectName("FaceLabel2_1")
        self.horizontalLayout_9.addWidget(self.FaceLabel2_1)
        self.verticalLayoutWidget_8 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_8.setGeometry(QtCore.QRect(200, 400, 141, 191))
        self.verticalLayoutWidget_8.setObjectName("verticalLayoutWidget_8")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_8)
        self.verticalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.TextLabel2_3 = QtWidgets.QLabel(self.verticalLayoutWidget_8)
        self.TextLabel2_3.setObjectName("TextLabel2_3")
        self.verticalLayout_11.addWidget(self.TextLabel2_3)
        self.verticalLayoutWidget_9 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_9.setGeometry(QtCore.QRect(200, 600, 141, 191))
        self.verticalLayoutWidget_9.setObjectName("verticalLayoutWidget_9")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_9)
        self.verticalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.TextLabel2_4 = QtWidgets.QLabel(self.verticalLayoutWidget_9)
        self.TextLabel2_4.setObjectName("TextLabel2_4")
        self.verticalLayout_12.addWidget(self.TextLabel2_4)
        self.horizontalLayoutWidget_10 = QtWidgets.QWidget(self.tab_2)
        self.horizontalLayoutWidget_10.setGeometry(QtCore.QRect(0, 600, 191, 191))
        self.horizontalLayoutWidget_10.setObjectName("horizontalLayoutWidget_10")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_10)
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.FaceLabel2_4 = QtWidgets.QLabel(self.horizontalLayoutWidget_10)
        self.FaceLabel2_4.setObjectName("FaceLabel2_4")
        self.horizontalLayout_10.addWidget(self.FaceLabel2_4)
        self.FaceTab.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayoutWidget_10 = QtWidgets.QWidget(self.tab_3)
        self.verticalLayoutWidget_10.setGeometry(QtCore.QRect(200, 200, 141, 191))
        self.verticalLayoutWidget_10.setObjectName("verticalLayoutWidget_10")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_10)
        self.verticalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.TextLabel3_2 = QtWidgets.QLabel(self.verticalLayoutWidget_10)
        self.TextLabel3_2.setObjectName("TextLabel3_2")
        self.verticalLayout_13.addWidget(self.TextLabel3_2)
        self.horizontalLayoutWidget_11 = QtWidgets.QWidget(self.tab_3)
        self.horizontalLayoutWidget_11.setGeometry(QtCore.QRect(0, 400, 191, 191))
        self.horizontalLayoutWidget_11.setObjectName("horizontalLayoutWidget_11")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_11)
        self.horizontalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.FaceLabel3_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_11)
        self.FaceLabel3_3.setObjectName("FaceLabel3_3")
        self.horizontalLayout_11.addWidget(self.FaceLabel3_3)
        self.verticalLayoutWidget_11 = QtWidgets.QWidget(self.tab_3)
        self.verticalLayoutWidget_11.setGeometry(QtCore.QRect(200, 0, 141, 191))
        self.verticalLayoutWidget_11.setObjectName("verticalLayoutWidget_11")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_11)
        self.verticalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.TextLabel3_1 = QtWidgets.QLabel(self.verticalLayoutWidget_11)
        self.TextLabel3_1.setObjectName("TextLabel3_1")
        self.verticalLayout_14.addWidget(self.TextLabel3_1)
        self.horizontalLayoutWidget_12 = QtWidgets.QWidget(self.tab_3)
        self.horizontalLayoutWidget_12.setGeometry(QtCore.QRect(0, 200, 191, 191))
        self.horizontalLayoutWidget_12.setObjectName("horizontalLayoutWidget_12")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_12)
        self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.FaceLabel3_2 = QtWidgets.QLabel(self.horizontalLayoutWidget_12)
        self.FaceLabel3_2.setObjectName("FaceLabel3_2")
        self.horizontalLayout_12.addWidget(self.FaceLabel3_2)
        self.horizontalLayoutWidget_13 = QtWidgets.QWidget(self.tab_3)
        self.horizontalLayoutWidget_13.setGeometry(QtCore.QRect(0, 0, 191, 191))
        self.horizontalLayoutWidget_13.setObjectName("horizontalLayoutWidget_13")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_13)
        self.horizontalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.FaceLabel3_1 = QtWidgets.QLabel(self.horizontalLayoutWidget_13)
        self.FaceLabel3_1.setObjectName("FaceLabel3_1")
        self.horizontalLayout_13.addWidget(self.FaceLabel3_1)
        self.verticalLayoutWidget_12 = QtWidgets.QWidget(self.tab_3)
        self.verticalLayoutWidget_12.setGeometry(QtCore.QRect(200, 400, 141, 191))
        self.verticalLayoutWidget_12.setObjectName("verticalLayoutWidget_12")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_12)
        self.verticalLayout_15.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.TextLabel3_3 = QtWidgets.QLabel(self.verticalLayoutWidget_12)
        self.TextLabel3_3.setObjectName("TextLabel3_3")
        self.verticalLayout_15.addWidget(self.TextLabel3_3)
        self.verticalLayoutWidget_13 = QtWidgets.QWidget(self.tab_3)
        self.verticalLayoutWidget_13.setGeometry(QtCore.QRect(200, 600, 141, 191))
        self.verticalLayoutWidget_13.setObjectName("verticalLayoutWidget_13")
        self.verticalLayout_16 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_13)
        self.verticalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.TextLabel3_4 = QtWidgets.QLabel(self.verticalLayoutWidget_13)
        self.TextLabel3_4.setObjectName("TextLabel3_4")
        self.verticalLayout_16.addWidget(self.TextLabel3_4)
        self.horizontalLayoutWidget_14 = QtWidgets.QWidget(self.tab_3)
        self.horizontalLayoutWidget_14.setGeometry(QtCore.QRect(0, 600, 191, 191))
        self.horizontalLayoutWidget_14.setObjectName("horizontalLayoutWidget_14")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_14)
        self.horizontalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.FaceLabel3_4 = QtWidgets.QLabel(self.horizontalLayoutWidget_14)
        self.FaceLabel3_4.setObjectName("FaceLabel3_4")
        self.horizontalLayout_14.addWidget(self.FaceLabel3_4)
        self.FaceTab.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.verticalLayoutWidget_14 = QtWidgets.QWidget(self.tab_4)
        self.verticalLayoutWidget_14.setGeometry(QtCore.QRect(200, 600, 141, 191))
        self.verticalLayoutWidget_14.setObjectName("verticalLayoutWidget_14")
        self.verticalLayout_17 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_14)
        self.verticalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_17.setObjectName("verticalLayout_17")
        self.TextLabel4_4 = QtWidgets.QLabel(self.verticalLayoutWidget_14)
        self.TextLabel4_4.setObjectName("TextLabel4_4")
        self.verticalLayout_17.addWidget(self.TextLabel4_4)
        self.verticalLayoutWidget_15 = QtWidgets.QWidget(self.tab_4)
        self.verticalLayoutWidget_15.setGeometry(QtCore.QRect(200, 0, 141, 191))
        self.verticalLayoutWidget_15.setObjectName("verticalLayoutWidget_15")
        self.verticalLayout_18 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_15)
        self.verticalLayout_18.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_18.setObjectName("verticalLayout_18")
        self.TextLabel4_1 = QtWidgets.QLabel(self.verticalLayoutWidget_15)
        self.TextLabel4_1.setObjectName("TextLabel4_1")
        self.verticalLayout_18.addWidget(self.TextLabel4_1)
        self.horizontalLayoutWidget_15 = QtWidgets.QWidget(self.tab_4)
        self.horizontalLayoutWidget_15.setGeometry(QtCore.QRect(0, 200, 191, 191))
        self.horizontalLayoutWidget_15.setObjectName("horizontalLayoutWidget_15")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_15)
        self.horizontalLayout_15.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.FaceLabel4_2 = QtWidgets.QLabel(self.horizontalLayoutWidget_15)
        self.FaceLabel4_2.setObjectName("FaceLabel4_2")
        self.horizontalLayout_15.addWidget(self.FaceLabel4_2)
        self.verticalLayoutWidget_16 = QtWidgets.QWidget(self.tab_4)
        self.verticalLayoutWidget_16.setGeometry(QtCore.QRect(200, 400, 141, 191))
        self.verticalLayoutWidget_16.setObjectName("verticalLayoutWidget_16")
        self.verticalLayout_19 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_16)
        self.verticalLayout_19.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.TextLabel4_3 = QtWidgets.QLabel(self.verticalLayoutWidget_16)
        self.TextLabel4_3.setObjectName("TextLabel4_3")
        self.verticalLayout_19.addWidget(self.TextLabel4_3)
        self.horizontalLayoutWidget_16 = QtWidgets.QWidget(self.tab_4)
        self.horizontalLayoutWidget_16.setGeometry(QtCore.QRect(0, 0, 191, 191))
        self.horizontalLayoutWidget_16.setObjectName("horizontalLayoutWidget_16")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_16)
        self.horizontalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.FaceLabel4_1 = QtWidgets.QLabel(self.horizontalLayoutWidget_16)
        self.FaceLabel4_1.setObjectName("FaceLabel4_1")
        self.horizontalLayout_16.addWidget(self.FaceLabel4_1)
        self.verticalLayoutWidget_17 = QtWidgets.QWidget(self.tab_4)
        self.verticalLayoutWidget_17.setGeometry(QtCore.QRect(200, 200, 141, 191))
        self.verticalLayoutWidget_17.setObjectName("verticalLayoutWidget_17")
        self.verticalLayout_20 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_17)
        self.verticalLayout_20.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_20.setObjectName("verticalLayout_20")
        self.TextLabel4_2 = QtWidgets.QLabel(self.verticalLayoutWidget_17)
        self.TextLabel4_2.setObjectName("TextLabel4_2")
        self.verticalLayout_20.addWidget(self.TextLabel4_2)
        self.horizontalLayoutWidget_17 = QtWidgets.QWidget(self.tab_4)
        self.horizontalLayoutWidget_17.setGeometry(QtCore.QRect(0, 400, 191, 191))
        self.horizontalLayoutWidget_17.setObjectName("horizontalLayoutWidget_17")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_17)
        self.horizontalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.FaceLabel4_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_17)
        self.FaceLabel4_3.setObjectName("FaceLabel4_3")
        self.horizontalLayout_17.addWidget(self.FaceLabel4_3)
        self.horizontalLayoutWidget_18 = QtWidgets.QWidget(self.tab_4)
        self.horizontalLayoutWidget_18.setGeometry(QtCore.QRect(0, 600, 191, 191))
        self.horizontalLayoutWidget_18.setObjectName("horizontalLayoutWidget_18")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_18)
        self.horizontalLayout_18.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.FaceLabel4_4 = QtWidgets.QLabel(self.horizontalLayoutWidget_18)
        self.FaceLabel4_4.setObjectName("FaceLabel4_4")
        self.horizontalLayout_18.addWidget(self.FaceLabel4_4)
        self.FaceTab.addTab(self.tab_4, "")
        self.TabLayout.addWidget(self.FaceTab)
        self.horizontalLayoutWidget_19 = QtWidgets.QWidget(self)
        self.horizontalLayoutWidget_19.setGeometry(QtCore.QRect(360, 20, 351, 191))
        self.horizontalLayoutWidget_19.setObjectName("horizontalLayoutWidget_19")
        self.LogoLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_19)
        self.LogoLayout.setContentsMargins(0, 0, 0, 0)
        self.LogoLayout.setObjectName("LogoLayout")
        self.LabLogoLabel = QtWidgets.QLabel(self.horizontalLayoutWidget_19)
        self.LabLogoLabel.setObjectName("LabLogoLabel")
        self.LogoLayout.addWidget(self.LabLogoLabel)
        self.SJTULogoLabel = QtWidgets.QLabel(self.horizontalLayoutWidget_19)
        self.SJTULogoLabel.setObjectName("SJTULogoLabel")
        self.LogoLayout.addWidget(self.SJTULogoLabel)
        self.FaceTab.setCurrentIndex(0)



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
        self.ac_detection = self._contextMenu.addAction('识别', self.RecognitionOn)
        self.ac_record = self._contextMenu.addAction('记录', self.Record)
        self.ac_Addface = self._contextMenu.addAction('添加新人脸',self.AddFace)
    def initAnimation(self):
        # 按钮动画
        self._animation = QPropertyAnimation(
            self._contextMenu, b'geometry', self,
            easingCurve=QEasingCurve.Linear, duration=300)
        # easingCurve 修改该变量可以实现不同的效果

    #定义信号槽
    def slot_init(self):

        self.timer_camera.timeout.connect(self.show_camera)
        #人脸识别算法完成后在右边的tab widget 中显示
        self.FaceThread.Bound_Name.connect(self.ShowInTab)
    def AddFace(self):
        # if self.timer_camera.isActive() == False:
        #     flag = self.cap.open(self.CAM_NUM)
        #     if flag == False:
        #         msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please check you have connected your camera", buttons=QtWidgets.QMessageBox.Ok,
        #                                         defaultButton=QtWidgets.QMessageBox.Ok)
        # else:
        img = self.image
        print('cap ok')
        self.AddFaceThread.SetImg(img)


    def InputName(self):
        pass
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
            self.MainCameraLabel.clear()
            self.ac_open_cama.setText('打开相机')
    #相机显示
    def show_camera(self):
        flag, self.image= self.cap.read()
        show = cv2.resize(self.image, (800, 600))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],QImage.Format_RGB888 )
        self.MainCameraLabel.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def RecognitionOn(self):
        if self.timer_camera.isActive()==False:
            msg = QtWidgets.QMessageBox.warning(self, u"warning", u"没有检测到摄像头", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)

        else:
            #启动识别算法线程
            self.image = self.cap.read()
            self.FaceThread.SetImg(self.image)
            self.Lastimg  = self.image

    def Record(self):
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

    # def button_wrtieface_click(self):
    #     if self.timer_camera.isActive() == False:
    #         msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please open your camara ", buttons=QtWidgets.QMessageBox.Ok,
    #                                             defaultButton=QtWidgets.QMessageBox.Ok)
    #     else:
    #         name,ok = QInputDialog.getText(self, "Your name ", "Your name",
    #                                         QLineEdit.Normal, self.nameLable.text())
    #         if(ok and (len(name)!=0)):
    #             add_new_face(self.image,name)



    def ShowInTab(self,bound0,bound1,bound2,bound3, name):
        self.face = self.Lastimg[bound1:bound1 + bound3,
                    bound0:bound0 + bound2]
        show = cv2.resize(self.face, (200,200))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.FaceLabel1_1.setPixmap(QtGui.QPixmap.fromImage(showImage))
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



app = QtWidgets.QApplication(sys.argv)
ui = Ui_MainWindow()
ui.show()
sys.exit(app.exec_())