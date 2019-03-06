from __future__ import print_function
import sys
from includes.pymysql.PyMySQL import PyMySQL
from PyQt5 import QtCore, QtGui,QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import qdarkstyle

#数据库操作窗口


class TableWidge(QWidget):
    def __init__(self):
        super(TableWidge, self).__init__()
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.resize(500,300)
        self.db = PyMySQL('localhost','root','Asd980517','WEININGFACE')
        self.people_size = self.db.get_all_info().__len__()
        self.model = QStandardItemModel(2,self.people_size)
        self.model.setHorizontalHeaderLabels(['姓名','最近到访时间'])

        for col in range(self.people_size):
            item = QStandardItem(self.db.get_all_name()[col])
            self.model.setItem(0,col,item)
            item = QStandardItem(str(self.db.get_all_time()[col]))
            self.model.setItem(1,col,item)

        self.tableView = QTableView()
        self.tableView.setModel(self.model)
        layout = QVBoxLayout()
        layout.addWidget(self.tableView)
        self.setLayout(layout)


class DBWidge(QWidget):
    def __init__(self,parent=None):
        super(DBWidge,self).__init__(parent)
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        self.db = PyMySQL('localhost','root','Asd980517','WEININGFACE')
        self.initButton()
        self.initSlot()
        self.table = TableWidge()
        self.show()

    def initButton(self):
        self.connectButton = QPushButton('连接数据库',self)
        self.connectButton.setGeometry(100,100,100,50)
        self.checkButton = QPushButton('查看数据',self)
        self.checkButton.setGeometry(300,300,100,50)

    def initSlot(self):
        self.connectButton.clicked.connect(self.connectDB)
        self.checkButton.clicked.connect(self.openView)

    def connectDB(self):
        try:
            self.db.connect()
            QMessageBox.about(self,'Connection','成功连接数据库')

        except:
            QMessageBox.about(self,'Connection','连接数据库失败')

    def openView(self):
        if self.table.isHidden():
            self.table.show()

        else:
            self.table.hide()


if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = DBWidge()
    ui.show()
    sys.exit(app.exec_())