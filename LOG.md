# 想法记录

## 资源与工具

qtdesigner 的.ui文件转换为 .py的方法: pyuic5 -o 1.py 1.ui 

 https://blog.csdn.net/captain811/article/details/79545847 Python项目的管理方法

http://www.cnblogs.com/archisama/ Pyqt5 的教程



## 想法



### Qmenu

 进入功能后重新更改菜单命名：

1.先使用clear ，然后重新addAction()

2.尝试 ActiveAction

-----------------

已完成，对某个Action 用 settext(str) 即可



### Multimedia调用摄像头





### JSON存储

也许可以用json来进行人脸特征向量的简单存储（如果不调用数据库的话）

简单实现程序与程序之间的数据共享



### Main 放在哪里



### Unit test 用于测试代码



### 程序打包

https://blog.csdn.net/appke846/article/details/80758925 介绍了一种打包的工具



### 使用Django进行web开发

https://blog.csdn.net/dao_wolf/article/details/79123599 Django的框架







### QThread 的使用





### 10.10

尝试使用 setStyleSheet()的方法，似乎没有用

采用Qpalette() 完成了背景的设置

（还需要设置tabwidget的背景）





### 10.11

改进了人脸识别的线程，但现在还有问题



### 10.13

人脸识别（签到）线程已经完全改好了，目前性能已经不错，接下来需要更加美化界面以及优化结构



### 11.12

清理代码|

怎么用python 的 os 库来完成文件夹的文件合并？



### 11.21

https://blog.csdn.net/weixin_39964552/article/details/82937144 利用Qgraphicview设置图片

https://blog.csdn.net/gusgao/article/details/48930229

https://blog.csdn.net/windscloud/article/details/79732014 python del的用法



### 12.5

#### Issue

1.动态人脸识别怎么实现? 应该以什么样的方式实现？ 



#### Solution

