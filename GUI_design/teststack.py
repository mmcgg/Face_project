# -*- coding: utf-8 -*-

# self implementation generated from reading ui file 'designstack.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_self(object):
    def setupUi(self, self):
        self.setObjectName("self")
        self.resize(993, 836)
        self.stackedWidget = QtWidgets.QStackedWidget(self)
        self.stackedWidget.setGeometry(QtCore.QRect(420, 40, 401, 701))
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.graphicsView = QtWidgets.QGraphicsView(self.page)
        self.graphicsView.setGeometry(QtCore.QRect(0, 10, 371, 681))
        self.graphicsView.setObjectName("graphicsView")
        self.stackedWidget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.label = QtWidgets.QLabel(self.page_2)
        self.label.setGeometry(QtCore.QRect(0, 0, 211, 211))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.page_2)
        self.label_2.setGeometry(QtCore.QRect(210, 10, 161, 91))
        self.label_2.setObjectName("label_2")
        self.stackedWidget.addWidget(self.page_2)

        self.retranslateUi(self)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self, self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("self", "self"))
        self.label.setText(_translate("self", "TextLabel"))
        self.label_2.setText(_translate("self", "TextLabel"))

