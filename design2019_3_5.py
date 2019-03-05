# -*- coding: utf-8 -*-

# self implementation generated from reading ui file './GUI_design/design_2019_3_5.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!
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
from includes.Face.matlab_cp2tself import get_similarity_transself_for_cv2
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
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
class Ui_self(object):
    def setupUi(self, self):
        self.setObjectName("self")
        self.resize(1083, 846)
        self.textBrowser = QtWidgets.QTextBrowser(self)
        self.textBrowser.setGeometry(QtCore.QRect(10, 650, 661, 151))
        self.textBrowser.setObjectName("textBrowser")
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
        self.camera_labe = QtWidgets.QLabel(self)
        self.camera_labe.setGeometry(QtCore.QRect(10, 90, 661, 551))
        self.camera_labe.setObjectName("camera_labe")
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
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)

        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self, self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("self", "self"))
        self.faceLabel1.setText(_translate("self", "TextLabel"))
        self.infoLabel3.setText(_translate("self", "TextLabel"))
        self.infoLabel1.setText(_translate("self", "TextLabel"))
        self.faceLabel3.setText(_translate("self", "TextLabel"))
        self.infoLabel2.setText(_translate("self", "TextLabel"))
        self.faceLabel2.setText(_translate("self", "TextLabel"))
        self.faceLabel4.setText(_translate("self", "TextLabel"))
        self.infoLabel4.setText(_translate("self", "TextLabel"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("self", "Tab 1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("self", "Tab 2"))
        self.camera_labe.setText(_translate("self", "TextLabel"))
        self.pushButton_4.setText(_translate("self", "PushButton"))
        self.pushButton_3.setText(_translate("self", "PushButton"))
        self.pushButton_2.setText(_translate("self", "PushButton"))
        self.pushButton.setText(_translate("self", "打开"))




if __name__=='__main__':
    app = QApplication(sys.argv)
    Mainwin = QMainWindow()
    ui = Ui_self()
    ui.setupUi(Mainwin)
    Mainwin.show()
    sys.exit(app.exec_())

