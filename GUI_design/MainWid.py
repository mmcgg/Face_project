# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design3.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1114, 861)
        self.horizontalLayoutWidget = QtWidgets.QWidget(Form)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 230, 731, 561))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.MainCameraLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.MainCameraLayout.setContentsMargins(0, 0, 0, 0)
        self.MainCameraLayout.setObjectName("MainCameraLayout")
        self.MainCameraLabel = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.MainCameraLabel.setObjectName("MainCameraLabel")
        self.MainCameraLayout.addWidget(self.MainCameraLabel)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(Form)
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
        self.horizontalLayoutWidget_19 = QtWidgets.QWidget(Form)
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

        self.retranslateUi(Form)
        self.FaceTab.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.MainCameraLabel.setText(_translate("Form", "TextLabel"))
        self.FaceLabel1_1.setText(_translate("Form", "TextLabel"))
        self.FaceLabel1_2.setText(_translate("Form", "TextLabel"))
        self.FaceLabel1_3.setText(_translate("Form", "TextLabel"))
        self.TextLabel1_1.setText(_translate("Form", "TextLabel"))
        self.TextLabel1_2.setText(_translate("Form", "TextLabel"))
        self.TextLabel1_3.setText(_translate("Form", "TextLabel"))
        self.FaceLabel1_4.setText(_translate("Form", "TextLabel"))
        self.TextLabel1_4.setText(_translate("Form", "TextLabel"))
        self.FaceTab.setTabText(self.FaceTab.indexOf(self.tab), _translate("Form", "1"))
        self.TextLabel2_2.setText(_translate("Form", "TextLabel"))
        self.FaceLabel2_3.setText(_translate("Form", "TextLabel"))
        self.TextLabel2_1.setText(_translate("Form", "TextLabel"))
        self.FaceLabel2_2.setText(_translate("Form", "TextLabel"))
        self.FaceLabel2_1.setText(_translate("Form", "TextLabel"))
        self.TextLabel2_3.setText(_translate("Form", "TextLabel"))
        self.TextLabel2_4.setText(_translate("Form", "TextLabel"))
        self.FaceLabel2_4.setText(_translate("Form", "TextLabel"))
        self.FaceTab.setTabText(self.FaceTab.indexOf(self.tab_2), _translate("Form", "2"))
        self.TextLabel3_2.setText(_translate("Form", "TextLabel"))
        self.FaceLabel3_3.setText(_translate("Form", "TextLabel"))
        self.TextLabel3_1.setText(_translate("Form", "TextLabel"))
        self.FaceLabel3_2.setText(_translate("Form", "TextLabel"))
        self.FaceLabel3_1.setText(_translate("Form", "TextLabel"))
        self.TextLabel3_3.setText(_translate("Form", "TextLabel"))
        self.TextLabel3_4.setText(_translate("Form", "TextLabel"))
        self.FaceLabel3_4.setText(_translate("Form", "TextLabel"))
        self.FaceTab.setTabText(self.FaceTab.indexOf(self.tab_3), _translate("Form", "3"))
        self.TextLabel4_4.setText(_translate("Form", "TextLabel"))
        self.TextLabel4_1.setText(_translate("Form", "TextLabel"))
        self.FaceLabel4_2.setText(_translate("Form", "TextLabel"))
        self.TextLabel4_3.setText(_translate("Form", "TextLabel"))
        self.FaceLabel4_1.setText(_translate("Form", "TextLabel"))
        self.TextLabel4_2.setText(_translate("Form", "TextLabel"))
        self.FaceLabel4_3.setText(_translate("Form", "TextLabel"))
        self.FaceLabel4_4.setText(_translate("Form", "TextLabel"))
        self.FaceTab.setTabText(self.FaceTab.indexOf(self.tab_4), _translate("Form", "4"))
        self.LabLogoLabel.setText(_translate("Form", "TextLabel"))
        self.SJTULogoLabel.setText(_translate("Form", "TextLabel"))

