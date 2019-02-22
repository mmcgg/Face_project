from __future__ import print_function
import sys
from includes.pymysql.PyMySQL import PyMySQL
from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import qdarkstyle

#数据库操作窗口

class DBWidge(QWidget):
    def __init__(self,parent=None):
        super(DBWidge,self).__init__(parent)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        self.db = PyMySQL('localhost','root','Asd980517','WEININGFACE')
        self.initButton()
        self.initSlot()

        self.show()

    def initButton(self):
        self.connectButton = QPushButton('连接数据库',self)
        self.connectButton.setGeometry(100,100,200,50)

    def initSlot(self):
        self.connectButton.clicked.connect(self.connectDB)

    def connectDB(self):
        try:
            self.db.connect()
            QMessageBox.about(self,'Connection','成功连接数据库')

        except:
            QMessageBox.about(self,'Connection','连接数据库失败')



if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = DBWidge()
    ui.show()
    sys.exit(app.exec_())