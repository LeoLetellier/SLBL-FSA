# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form_comp_disp.ui'
##
## Created by: Qt User Interface Compiler version 6.5.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QGroupBox,
    QLabel, QPushButton, QSizePolicy, QWidget)

class Ui_Form(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(300, 400)
        font = QFont()
        font.setFamilies([u"Bahnschrift"])
        font.setPointSize(12)
        Form.setFont(font)
        self.comp_disp_title = QLabel(Form)
        self.comp_disp_title.setObjectName(u"comp_disp_title")
        self.comp_disp_title.setGeometry(QRect(10, -10, 281, 81))
        font1 = QFont()
        font1.setFamilies([u"Bahnschrift SemiBold"])
        font1.setPointSize(18)
        font1.setBold(True)
        self.comp_disp_title.setFont(font1)
        self.groupBox_cross_sec = QGroupBox(Form)
        self.groupBox_cross_sec.setObjectName(u"groupBox_cross_sec")
        self.groupBox_cross_sec.setGeometry(QRect(20, 70, 261, 61))
        self.label = QLabel(self.groupBox_cross_sec)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(11, 31, 64, 16))
        self.doubleSpinBox_alpha = QDoubleSpinBox(self.groupBox_cross_sec)
        self.doubleSpinBox_alpha.setObjectName(u"doubleSpinBox_alpha")
        self.doubleSpinBox_alpha.setGeometry(QRect(133, 31, 80, 21))
        self.doubleSpinBox_alpha.setMinimum(-360.000000000000000)
        self.doubleSpinBox_alpha.setMaximum(360.000000000000000)
        self.groupBox_LOS = QGroupBox(Form)
        self.groupBox_LOS.setObjectName(u"groupBox_LOS")
        self.groupBox_LOS.setGeometry(QRect(20, 150, 261, 91))
        self.label_2 = QLabel(self.groupBox_LOS)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(11, 32, 62, 16))
        self.doubleSpinBox_theta = QDoubleSpinBox(self.groupBox_LOS)
        self.doubleSpinBox_theta.setObjectName(u"doubleSpinBox_theta")
        self.doubleSpinBox_theta.setGeometry(QRect(133, 32, 80, 21))
        self.doubleSpinBox_theta.setMinimum(-360.000000000000000)
        self.doubleSpinBox_theta.setMaximum(360.000000000000000)
        self.doubleSpinBox_theta.setValue(35.000000000000000)
        self.label_3 = QLabel(self.groupBox_LOS)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(10, 58, 61, 16))
        self.doubleSpinBox_delta = QDoubleSpinBox(self.groupBox_LOS)
        self.doubleSpinBox_delta.setObjectName(u"doubleSpinBox_delta")
        self.doubleSpinBox_delta.setGeometry(QRect(133, 58, 80, 21))
        self.doubleSpinBox_delta.setMinimum(-360.000000000000000)
        self.doubleSpinBox_delta.setMaximum(360.000000000000000)
        self.doubleSpinBox_delta.setValue(285.000000000000000)
        self.btn_C_k = QPushButton(Form)
        self.btn_C_k.setObjectName(u"btn_C_k")
        self.btn_C_k.setGeometry(QRect(0, 280, 300, 41))
        self.btn_C_rkms = QPushButton(Form)
        self.btn_C_rkms.setObjectName(u"btn_C_rkms")
        self.btn_C_rkms.setGeometry(QRect(0, 330, 300, 41))
        self.do_def = QCheckBox(Form)
        self.do_def.setObjectName(u"do_def")
        self.do_def.setGeometry(QRect(25, 250, 251, 21))

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.comp_disp_title.setText(QCoreApplication.translate("Form", u"Compare displacement", None))
        self.groupBox_cross_sec.setTitle(QCoreApplication.translate("Form", u"Cross-section", None))
        self.label.setText(QCoreApplication.translate("Form", u"Alpha", None))
        self.doubleSpinBox_alpha.setSuffix(QCoreApplication.translate("Form", u"\u00b0", None))
        self.groupBox_LOS.setTitle(QCoreApplication.translate("Form", u"Data LOS", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"Theta", None))
        self.doubleSpinBox_theta.setSuffix(QCoreApplication.translate("Form", u"\u00b0", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"Delta", None))
        self.doubleSpinBox_delta.setSuffix(QCoreApplication.translate("Form", u"\u00b0", None))
        self.btn_C_k.setText(QCoreApplication.translate("Form", u"Compute model (user ratios)", None))
        self.btn_C_rkms.setText(QCoreApplication.translate("Form", u"Compute model (RMSE least square)", None))
        self.do_def.setText(QCoreApplication.translate("Form", u"Use model with deformation", None))
    # retranslateUi

