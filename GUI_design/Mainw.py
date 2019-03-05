
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, QThreadPool
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtWidgets import QApplication, QLineEdit, QInputDialog, QGridLayout, QLabel, QPushButton, QFrame, QWidget,QMenu
import cv2
class Mainw(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1106, 850)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(10, 110, 681, 551))
        self.widget.setObjectName("widget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.widget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 681, 551))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.CameraLabel = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.CameraLabel.setObjectName("CameraLabel")
        self.horizontalLayout.addWidget(self.CameraLabel)
        self.showface = QtWidgets.QTabWidget(self.centralwidget)
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
        self.showface.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.listView_2 = QtWidgets.QListView(self.tab_2)
        self.listView_2.setGeometry(QtCore.QRect(20, 20, 299, 609))
        self.listView_2.setObjectName("listView_2")
        self.showface.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1106, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        self.showface.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        #0.其它子对象创建
        self._contextMenu  = QMenu(self.CameraLabel)
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        #1.右键下拉菜单的创建
        self.initMenu()

        #2.初始化信号与信号槽
        self.slot_init()


        MainWindow.show()
    def contextMenuEvent(self, event):
        pos = event.globalPos()
        size = self._contextMenu.sizeHint()
        x, y, w, h = pos.x(), pos.y(), size.width(), size.height()
        self._animation.stop()
        self._animation.setStartValue(QRect(x, y, 0, 0))
        self._animation.setEndValue(QRect(x, y, w, h))
        self._animation.start()
        self._contextMenu.exec_(event.globalPos())

    def slot_init(self):
        self.timer_camera.timeout.connect(self.show_camera)
    def initMenu(self):
        self.ac_open_cama = self._contextMenu.addAction('打开相机', self.CameraOperation)
        self.ac_recognition = self._contextMenu.addAction('识别', self.RecognitionOperation)
        self.ac_record = self._contextMenu.addAction('记录', self.RecordOperation)
        self.ac_Addface = self._contextMenu.addAction('添加人脸',self.Addface)
    def initAnimation(self):
        # 按钮动画
        self._animation = QPropertyAnimation(
            self._contextMenu, b'geometry', self,
            easingCurve=QEasingCurve.Linear, duration=300)
        # easingCurve 修改该变量可以实现不同的效果

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.CameraLabel.setText(_translate("MainWindow", "TextLabel"))
        self.showface.setTabText(self.showface.indexOf(self.tab), _translate("MainWindow", "Tab 1"))
        self.showface.setTabText(self.showface.indexOf(self.tab_2), _translate("MainWindow", "Tab 2"))

    def show_camera(self):
        flag, self.image= self.cap.read()
        show = cv2.resize(self.image, (800,600))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.selfat_RGB888)
        self.CameraLabel.setPixmap(QtGui.QPixmap.fromImage(showImage))

    # 打开相机操作
    def CameraOperation(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请先打开相机",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)

            else:
                self.timer_camera.start(50)
                self.ac_open_cama.setText('关闭相机')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.CameraLabel.clear()
            self.ac_open_cama.setText('打开相机')

    #识别
    def RecognitionOperation(self):
        pass
    #添加人脸
    def Addface(self):
        pass

    #记录
    def RecordOperation(self):
        pass
app = QApplication(sys.argv)
UI = QtWidgets.QMainWindow()
ui = Mainw()
ui.setupUi(UI)


exit(app.exec_())