# This Python file uses the following encoding: utf-8
import sys

from PySide6.QtWidgets import QApplication, QMainWindow

from ui_form import Ui_MainWindow
from ui_connection import UiConnect

from LandslideAreanSLBL import *


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.area = LandslideArea()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.connect = UiConnect(self.ui, self.area)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
