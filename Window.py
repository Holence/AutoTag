# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Window.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Window(object):
    def setupUi(self, Window):
        if not Window.objectName():
            Window.setObjectName(u"Window")
        Window.resize(1109, 613)
        self.horizontalLayout_2 = QHBoxLayout(Window)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.plainTextEdit_vec = QPlainTextEdit(Window)
        self.plainTextEdit_vec.setObjectName(u"plainTextEdit_vec")

        self.horizontalLayout_2.addWidget(self.plainTextEdit_vec)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.lineEdit_train_text = QLineEdit(Window)
        self.lineEdit_train_text.setObjectName(u"lineEdit_train_text")
        sizePolicy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_train_text.sizePolicy().hasHeightForWidth())
        self.lineEdit_train_text.setSizePolicy(sizePolicy)

        self.horizontalLayout.addWidget(self.lineEdit_train_text)

        self.lineEdit_train_tag = QLineEdit(Window)
        self.lineEdit_train_tag.setObjectName(u"lineEdit_train_tag")
        sizePolicy1 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lineEdit_train_tag.sizePolicy().hasHeightForWidth())
        self.lineEdit_train_tag.setSizePolicy(sizePolicy1)

        self.horizontalLayout.addWidget(self.lineEdit_train_tag)


        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.pushButton_forward = QPushButton(Window)
        self.pushButton_forward.setObjectName(u"pushButton_forward")

        self.verticalLayout.addWidget(self.pushButton_forward)

        self.pushButton_backward = QPushButton(Window)
        self.pushButton_backward.setObjectName(u"pushButton_backward")

        self.verticalLayout.addWidget(self.pushButton_backward)


        self.gridLayout.addLayout(self.verticalLayout, 0, 1, 1, 1)

        self.lineEdit_pred_text = QLineEdit(Window)
        self.lineEdit_pred_text.setObjectName(u"lineEdit_pred_text")

        self.gridLayout.addWidget(self.lineEdit_pred_text, 1, 0, 1, 1)

        self.pushButton_pred = QPushButton(Window)
        self.pushButton_pred.setObjectName(u"pushButton_pred")

        self.gridLayout.addWidget(self.pushButton_pred, 1, 1, 1, 1)

        self.plainTextEdit_pred_tag = QPlainTextEdit(Window)
        self.plainTextEdit_pred_tag.setObjectName(u"plainTextEdit_pred_tag")

        self.gridLayout.addWidget(self.plainTextEdit_pred_tag, 2, 0, 1, 1)


        self.horizontalLayout_2.addLayout(self.gridLayout)


        self.retranslateUi(Window)

        QMetaObject.connectSlotsByName(Window)
    # setupUi

    def retranslateUi(self, Window):
        Window.setWindowTitle(QCoreApplication.translate("Window", u"Form", None))
        self.pushButton_forward.setText(QCoreApplication.translate("Window", u" Forward(Train) ", None))
        self.pushButton_backward.setText(QCoreApplication.translate("Window", u" Backward(De-Train) ", None))
        self.pushButton_pred.setText(QCoreApplication.translate("Window", u" Predict ", None))
    # retranslateUi

