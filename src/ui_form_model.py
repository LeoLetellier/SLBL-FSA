# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form_model.ui'
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
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QGroupBox, QHeaderView,
    QLabel, QPushButton, QRadioButton, QSizePolicy,
    QTableWidget, QTableWidgetItem, QWidget)

class Ui_Form(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(640, 480)
        self.table = QTableWidget(Form)
        self.table.setObjectName(u"table")
        self.table.setGeometry(QRect(260, 0, 381, 481))
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.load_a = QPushButton(Form)
        self.load_a.setObjectName(u"load_a")
        self.load_a.setGeometry(QRect(80, 10, 100, 25))
        self.generate_table = QPushButton(Form)
        self.generate_table.setObjectName(u"generate_table")
        self.generate_table.setGeometry(QRect(20, 180, 221, 41))
        self.apply_a = QPushButton(Form)
        self.apply_a.setObjectName(u"apply_a")
        self.apply_a.setGeometry(QRect(50, 380, 151, 51))
        self.save_a = QPushButton(Form)
        self.save_a.setObjectName(u"save_a")
        self.save_a.setGeometry(QRect(80, 440, 100, 25))
        self.label_3 = QLabel(Form)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(20, 270, 171, 41))
        self.groupBox = QGroupBox(Form)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(30, 40, 211, 121))
        self.an = QDoubleSpinBox(self.groupBox)
        self.an.setObjectName(u"an")
        self.an.setGeometry(QRect(130, 80, 62, 22))
        self.an.setMinimum(-100.000000000000000)
        self.an.setMaximum(0.000000000000000)
        self.an.setValue(-1.000000000000000)
        self.ap = QDoubleSpinBox(self.groupBox)
        self.ap.setObjectName(u"ap")
        self.ap.setGeometry(QRect(130, 40, 62, 22))
        self.ap.setMaximum(100.000000000000000)
        self.ap.setValue(1.000000000000000)
        self.do_an = QRadioButton(self.groupBox)
        self.do_an.setObjectName(u"do_an")
        self.do_an.setGeometry(QRect(10, 80, 111, 18))
        self.do_ap = QRadioButton(self.groupBox)
        self.do_ap.setObjectName(u"do_ap")
        self.do_ap.setGeometry(QRect(10, 40, 111, 18))
        self.do_ap.setChecked(True)
        self.np = QLabel(Form)
        self.np.setObjectName(u"np")
        self.np.setGeometry(QRect(170, 270, 91, 41))
        self.label_5 = QLabel(Form)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(60, 320, 151, 41))
        self.vv = QLabel(Form)
        self.vv.setObjectName(u"vv")
        self.vv.setGeometry(QRect(170, 320, 91, 41))
        self.correct_table = QPushButton(Form)
        self.correct_table.setObjectName(u"correct_table")
        self.correct_table.setGeometry(QRect(20, 230, 221, 41))

        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.load_a.setText(QCoreApplication.translate("Form", u"Load from file", None))
        self.generate_table.setText(QCoreApplication.translate("Form", u"Generate table", None))
        self.apply_a.setText(QCoreApplication.translate("Form", u"Apply table on model", None))
        self.save_a.setText(QCoreApplication.translate("Form", u"Save to file", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"Neutral point position (m): ", None))
        self.groupBox.setTitle(QCoreApplication.translate("Form", u"Height correction", None))
        self.do_an.setText(QCoreApplication.translate("Form", u"negative (a-)", None))
        self.do_ap.setText(QCoreApplication.translate("Form", u"positive (a+)", None))
        self.np.setText(QCoreApplication.translate("Form", u"_", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"Volume variation:", None))
        self.vv.setText(QCoreApplication.translate("Form", u"_", None))
        self.correct_table.setText(QCoreApplication.translate("Form", u"Correct table (custom values)", None))
    # retranslateUi

