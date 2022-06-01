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
        Window.resize(887, 445)
        self.horizontalLayout = QHBoxLayout(Window)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label_3 = QLabel(Window)
        self.label_3.setObjectName(u"label_3")

        self.verticalLayout.addWidget(self.label_3)

        self.plainTextEdit_train_text = QPlainTextEdit(Window)
        self.plainTextEdit_train_text.setObjectName(u"plainTextEdit_train_text")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plainTextEdit_train_text.sizePolicy().hasHeightForWidth())
        self.plainTextEdit_train_text.setSizePolicy(sizePolicy)
        self.plainTextEdit_train_text.setMinimumSize(QSize(250, 350))

        self.verticalLayout.addWidget(self.plainTextEdit_train_text)

        self.lineEdit_train_tag = QLineEdit(Window)
        self.lineEdit_train_tag.setObjectName(u"lineEdit_train_tag")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lineEdit_train_tag.sizePolicy().hasHeightForWidth())
        self.lineEdit_train_tag.setSizePolicy(sizePolicy1)

        self.verticalLayout.addWidget(self.lineEdit_train_tag)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.pushButton_forward = QPushButton(Window)
        self.pushButton_forward.setObjectName(u"pushButton_forward")
        sizePolicy1.setHeightForWidth(self.pushButton_forward.sizePolicy().hasHeightForWidth())
        self.pushButton_forward.setSizePolicy(sizePolicy1)

        self.horizontalLayout_2.addWidget(self.pushButton_forward)

        self.pushButton_backward = QPushButton(Window)
        self.pushButton_backward.setObjectName(u"pushButton_backward")
        sizePolicy1.setHeightForWidth(self.pushButton_backward.sizePolicy().hasHeightForWidth())
        self.pushButton_backward.setSizePolicy(sizePolicy1)

        self.horizontalLayout_2.addWidget(self.pushButton_backward)


        self.verticalLayout.addLayout(self.horizontalLayout_2)


        self.horizontalLayout.addLayout(self.verticalLayout)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.splitter = QSplitter(Window)
        self.splitter.setObjectName(u"splitter")
        self.splitter.setOrientation(Qt.Horizontal)
        self.layoutWidget = QWidget(self.splitter)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.verticalLayout_4 = QVBoxLayout(self.layoutWidget)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.layoutWidget)
        self.label.setObjectName(u"label")

        self.verticalLayout_4.addWidget(self.label)

        self.plainTextEdit_pred_text = QPlainTextEdit(self.layoutWidget)
        self.plainTextEdit_pred_text.setObjectName(u"plainTextEdit_pred_text")
        sizePolicy2 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.plainTextEdit_pred_text.sizePolicy().hasHeightForWidth())
        self.plainTextEdit_pred_text.setSizePolicy(sizePolicy2)

        self.verticalLayout_4.addWidget(self.plainTextEdit_pred_text)

        self.splitter.addWidget(self.layoutWidget)
        self.layoutWidget_2 = QWidget(self.splitter)
        self.layoutWidget_2.setObjectName(u"layoutWidget_2")
        self.verticalLayout_5 = QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setSpacing(12)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_2 = QLabel(self.layoutWidget_2)
        self.label_2.setObjectName(u"label_2")
        sizePolicy3 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy3)

        self.horizontalLayout_3.addWidget(self.label_2)

        self.label_acc = QLabel(self.layoutWidget_2)
        self.label_acc.setObjectName(u"label_acc")

        self.horizontalLayout_3.addWidget(self.label_acc)


        self.verticalLayout_5.addLayout(self.horizontalLayout_3)

        self.textBrowser_res = QTextBrowser(self.layoutWidget_2)
        self.textBrowser_res.setObjectName(u"textBrowser_res")
        sizePolicy2.setHeightForWidth(self.textBrowser_res.sizePolicy().hasHeightForWidth())
        self.textBrowser_res.setSizePolicy(sizePolicy2)

        self.verticalLayout_5.addWidget(self.textBrowser_res)

        self.splitter.addWidget(self.layoutWidget_2)

        self.verticalLayout_2.addWidget(self.splitter)

        self.pushButton_pred = QPushButton(Window)
        self.pushButton_pred.setObjectName(u"pushButton_pred")
        sizePolicy1.setHeightForWidth(self.pushButton_pred.sizePolicy().hasHeightForWidth())
        self.pushButton_pred.setSizePolicy(sizePolicy1)
        self.pushButton_pred.setMinimumSize(QSize(500, 0))

        self.verticalLayout_2.addWidget(self.pushButton_pred)


        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_3.addItem(self.verticalSpacer)

        self.pushButton_load_corpus = QPushButton(Window)
        self.pushButton_load_corpus.setObjectName(u"pushButton_load_corpus")
        sizePolicy1.setHeightForWidth(self.pushButton_load_corpus.sizePolicy().hasHeightForWidth())
        self.pushButton_load_corpus.setSizePolicy(sizePolicy1)

        self.verticalLayout_3.addWidget(self.pushButton_load_corpus)

        self.pushButton_normal_train = QPushButton(Window)
        self.pushButton_normal_train.setObjectName(u"pushButton_normal_train")
        sizePolicy1.setHeightForWidth(self.pushButton_normal_train.sizePolicy().hasHeightForWidth())
        self.pushButton_normal_train.setSizePolicy(sizePolicy1)

        self.verticalLayout_3.addWidget(self.pushButton_normal_train)

        self.pushButton_continual_train = QPushButton(Window)
        self.pushButton_continual_train.setObjectName(u"pushButton_continual_train")
        sizePolicy1.setHeightForWidth(self.pushButton_continual_train.sizePolicy().hasHeightForWidth())
        self.pushButton_continual_train.setSizePolicy(sizePolicy1)

        self.verticalLayout_3.addWidget(self.pushButton_continual_train)

        self.pushButton_plot = QPushButton(Window)
        self.pushButton_plot.setObjectName(u"pushButton_plot")
        sizePolicy1.setHeightForWidth(self.pushButton_plot.sizePolicy().hasHeightForWidth())
        self.pushButton_plot.setSizePolicy(sizePolicy1)

        self.verticalLayout_3.addWidget(self.pushButton_plot)

        self.pushButton_plot2D = QPushButton(Window)
        self.pushButton_plot2D.setObjectName(u"pushButton_plot2D")
        sizePolicy1.setHeightForWidth(self.pushButton_plot2D.sizePolicy().hasHeightForWidth())
        self.pushButton_plot2D.setSizePolicy(sizePolicy1)

        self.verticalLayout_3.addWidget(self.pushButton_plot2D)

        self.pushButton_eval = QPushButton(Window)
        self.pushButton_eval.setObjectName(u"pushButton_eval")
        sizePolicy1.setHeightForWidth(self.pushButton_eval.sizePolicy().hasHeightForWidth())
        self.pushButton_eval.setSizePolicy(sizePolicy1)

        self.verticalLayout_3.addWidget(self.pushButton_eval)

        self.pushButton_save = QPushButton(Window)
        self.pushButton_save.setObjectName(u"pushButton_save")
        sizePolicy1.setHeightForWidth(self.pushButton_save.sizePolicy().hasHeightForWidth())
        self.pushButton_save.setSizePolicy(sizePolicy1)

        self.verticalLayout_3.addWidget(self.pushButton_save)


        self.horizontalLayout.addLayout(self.verticalLayout_3)

        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 10)
        self.horizontalLayout.setStretch(2, 1)

        self.retranslateUi(Window)

        QMetaObject.connectSlotsByName(Window)
    # setupUi

    def retranslateUi(self, Window):
        Window.setWindowTitle(QCoreApplication.translate("Window", u"Form", None))
        self.label_3.setText(QCoreApplication.translate("Window", u"Train Text", None))
        self.pushButton_forward.setText(QCoreApplication.translate("Window", u" Train ", None))
        self.pushButton_backward.setText(QCoreApplication.translate("Window", u" De-Train ", None))
        self.label.setText(QCoreApplication.translate("Window", u"Predict Text", None))
        self.label_2.setText(QCoreApplication.translate("Window", u"Result", None))
        self.label_acc.setText("")
        self.pushButton_pred.setText(QCoreApplication.translate("Window", u" Predict ", None))
        self.pushButton_load_corpus.setText(QCoreApplication.translate("Window", u" Load Corpus ", None))
        self.pushButton_normal_train.setText(QCoreApplication.translate("Window", u" Normal Train ", None))
        self.pushButton_continual_train.setText(QCoreApplication.translate("Window", u" Continual Train ", None))
        self.pushButton_plot.setText(QCoreApplication.translate("Window", u" Plot ", None))
        self.pushButton_plot2D.setText(QCoreApplication.translate("Window", u" Plot 2D ", None))
        self.pushButton_eval.setText(QCoreApplication.translate("Window", u" Evaluate ", None))
        self.pushButton_save.setText(QCoreApplication.translate("Window", u" Save Model ", None))
    # retranslateUi

