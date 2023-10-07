"""
Python file handling the link between python variables and processing, and the Qt interface
"""

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QFileDialog, QVBoxLayout, QMessageBox, QTableWidgetItem)

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from LandslideAreanSLBL import *
from SlblToolKit import *
from project_manager import *
from math import floor

from ui_form_slbl import Ui_Form as Ui_slbl
from ui_form_comp_disp import Ui_Form as Ui_comp_disp
from ui_form_model import Ui_Form as Ui_model


class UiConnect():
    def __init__(self, other, area: LandslideArea):
        self.area = area
        self.project = ProjectManager()
        self.other = other
        self.ui_slbl = None
        self.ui_def_model = None
        self.ui_comp_disp = None
        self.graph_is_setup = False

        self.initialize_available_slides()

        self.slide_selected = "slide1"
        self.dv_slide_selected = "slide1"
        self.dm_slide_selected = ["slide1"]

        # HOME buttons
        other.btn_load_dem.clicked.connect(self.load_dem)
        other.load_project.clicked.connect(self.load_project)
        other.save_project.clicked.connect(self.save_project)
        other.btn_docs.clicked.connect(self.help_docs)
        other.btn_file_format.clicked.connect(self.help_file_format)
        other.btn_about.clicked.connect(self.help_about)

        # SLBL buttons
        other.btn_load_slbl.clicked.connect(self.load_slbl)
        other.btn_compute_slbl.clicked.connect(self.compute_slbl)
        other.btn_graph_slbl.clicked.connect(self.graph_slbl)
        other.btn_clear_graph_slbl.clicked.connect(self.graph_clear)
        other.btn_save_slbl.clicked.connect(self.save_slbl)

        # DISP VEC buttons
        other.btn_lin_prof.clicked.connect(self.apply_linear_profile)
        other.btn_load_disp_vec.clicked.connect(self.load_disp_vec)
        other.btn_compute_disp_proc.clicked.connect(self.compute_disp)
        other.btn_graph_disp_vec.clicked.connect(self.graph_disp_vec)
        other.btn_clear_graph_disp_vec.clicked.connect(self.graph_clear)
        other.btn_save_disp_vec.clicked.connect(self.save_disp_vec)

        # COMP DISP buttons
        other.btn_load_disp_data.clicked.connect(self.load_disp_data)
        other.btn_def_model.clicked.connect(self.deformation_model)
        other.btn_compute_model.clicked.connect(self.compute_model)
        other.btn_graph_comp.clicked.connect(self.graph_comp)
        other.btn_graph_model.clicked.connect(self.graph_model)
        other.btn_graph_corr.clicked.connect(self.graph_corr)
        other.btn_clear_comp_disp.clicked.connect(self.graph_clear)
        other.btn_save_comp_disp.clicked.connect(self.save_model)

    def initialize_available_slides(self):
        slide_names = ['slide' + str(k) for k in range(1, 7)]
        for name in slide_names:
            self.area.access_slide(name)

    def update_slide_selected(self):
        inputs = [self.other.slbl1, self.other.slbl2, self.other.slbl3, self.other.slbl4, self.other.slbl5,
                  self.other.slbl6]
        for selection in range(len(inputs)):
            if inputs[selection].isChecked():
                self.slide_selected = "slide" + str(selection + 1)
                break

    def update_dv_slide_selected(self):
        inputs = [self.other.dv_slbl1, self.other.dv_slbl2, self.other.dv_slbl3, self.other.dv_slbl4,
                  self.other.dv_slbl5, self.other.dv_slbl6]
        for selection in range(len(inputs)):
            if inputs[selection].isChecked():
                self.dv_slide_selected = "slide" + str(selection + 1)
                break
        print("slide selected: ", self.dv_slide_selected)

    def update_dm_slide_selected(self):
        inputs = [self.other.dm_slbl1, self.other.dm_slbl2, self.other.dm_slbl3, self.other.dm_slbl4,
                  self.other.dm_slbl5, self.other.dm_slbl6]
        ratios = [self.other.ratio_slbl1, self.other.ratio_slbl2, self.other.ratio_slbl3, self.other.ratio_slbl4,
                  self.other.ratio_slbl5, self.other.ratio_slbl6]
        self.dm_slide_selected = []
        for selection in range(len(inputs)):
            if inputs[selection].isChecked():
                self.dm_slide_selected.append("slide" + str(selection + 1))
                self.area.slides[self.dm_slide_selected[-1]].disp_ratio = ratios[selection].value()

    def update_axis(self, check_box):
        if check_box.isChecked():
            return 'equal'
        return 'auto'

    @staticmethod
    def open_file(rd_fct):
        dialog = QFileDialog()
        file = dialog.getOpenFileName(filter="*.csv; *.xls; *.xlsx; *.txt; *.asc")  # file : tuple (path.ext, file type)
        if file[0] != "":  # No error when closed with no file selected
            rd_fct(file[0])
            return file[0]
        return None

    @staticmethod
    def save_file(sv_fct):
        dialog = QFileDialog()
        file = dialog.getSaveFileName(filter="*.csv")
        if file[0] != '':
            sv_fct(file[0])
            return file[0]
        return None

    def load_dem(self):
        file = self.open_file(self.area.load_dem)
        if file is not None:  # No error when closed with no file selected
            self.area.dem_path = file
            self.other.x_min.setMinimum(self.area.x.min())
            self.other.x_min.setMaximum(self.area.x.max())
            self.other.x_max.setMinimum(self.area.x.min())
            self.other.x_max.setMaximum(self.area.x.max())

    def load_slbl(self):
        self.update_slide_selected()
        file = self.open_file(self.area.slides[self.slide_selected].slbl.load)
        self.area.slides[self.slide_selected].slide_path = file
        self.area.slides[self.slide_selected].method = "PATH"

    def load_disp_vec(self):
        self.update_dv_slide_selected()
        self.open_file(self.area.slides[self.dv_slide_selected].block_disp.load)

    def load_disp_data(self):
        file = self.open_file(self.area.dc.load)
        self.area.dc.disp_path = file

    def save_slbl(self):
        self.update_slide_selected()
        self.save_file(self.area.slides[self.slide_selected].slbl.save_computed)

    def save_disp_vec(self):
        self.update_dv_slide_selected()
        self.save_file(self.area.slides[self.dv_slide_selected].block_disp.save)
        print("save disp vec from slide: ", self.dv_slide_selected)

    def save_model(self):
        self.save_file(self.area.model.save)

    def load_project(self):
        self.project = ProjectManager()
        dialog = QFileDialog()
        file = dialog.getOpenFileName(filter="*.project")
        if file[0] != "":
            self.project.load_project(self.area, file[0])

    def save_project(self):
        self.project = ProjectManager()
        dialog = QFileDialog()
        file = dialog.getSaveFileName(filter="*.project")
        if file[0] != '':
            self.project.save_project(self.area, file[0])

    def apply_linear_profile(self):
        if self.area.slides[self.dv_slide_selected].block_disp.disp_vec is None:
            msg = QMessageBox()
            msg.setText("No displacement vectors!")
            msg.setInformativeText("Compute the corresponding displacement vectors for " + self.dv_slide_selected)
            msg.exec()
        else:
            rl = self.other.r_left.value()
            rr = self.other.r_right.value()
            sum = rl + rr
            self.area.slides[self.dv_slide_selected].lin_value_left = rl / sum
            self.area.slides[self.dv_slide_selected].lin_value_right = rr / sum
            self.area.slides[self.dv_slide_selected].linear_profile_on_bd()

    def compute_slbl(self):
        self.update_slide_selected()
        if self.area.x is None:
            msg = QMessageBox()
            msg.setText("No DEM!")
            msg.setInformativeText("Please load a DEM")
            msg.exec()
        else:
            if self.ui_slbl is None:
                self.ui_slbl = Ui_slbl()
            self.ui_slbl_connect = UiConnectSLBL(self, self.ui_slbl, self.area)
            self.ui_slbl.show()

    def compute_disp(self):
        self.update_dv_slide_selected()
        if self.area.x is None or self.area.slides[self.dv_slide_selected].slbl.slbl_surf is None:
            msg = QMessageBox()
            msg.setText("Please load necessary data")
            msg.setInformativeText("Data: DEM, SLBL (input or computed)")
            msg.exec()
        else:
            if (self.area.model.n_vec, self.area.model.start_pnt, self.area.model.stop_pnt) != (
                    self.other.spinBox_n.value(), meter_2_idx(self.other.x_min.value(), self.area.x),
                    meter_2_idx(self.other.x_max.value(), self.area.x)):
                self.area.model.reset_model()
                for name in ['slide' + str(k) for k in range(1, 7)]:
                    self.area.slides[name].reset_bd()
                self.area.model.n_vec = self.other.spinBox_n.value()
                self.area.model.start_pnt = meter_2_idx(self.other.x_min.value(), self.area.x)
                self.area.model.stop_pnt = meter_2_idx(self.other.x_max.value(), self.area.x)
                self.area.model.slope_direction = np.sign(
                    (self.area.z[-1] - self.area.z[0]) / (self.area.x[-1] - self.area.x[0]))
            self.area.slides[self.dv_slide_selected].block_disp.compute_disp_vectors()
            if self.area.slides[self.dv_slide_selected].block_disp.test_crossing() is not True:
                msg = QMessageBox()
                msg.setText("WARNING: Block lines are intersecting themselves!")
                msg.setInformativeText("Please consider decreasing the number of vectors or using a smoother SLBL.")
                msg.exec()

    def deformation_model(self):
        if self.area.dc.disp_model is None:
            msg = QMessageBox()
            msg.setText("No Model!")
            msg.setInformativeText("Compute a model beforehand")
            msg.exec()
        else:
            if self.ui_def_model is None:
                self.ui_def_model = Ui_model()
            self.ui_def_model_connect = UiConnectDefModel(self, self.ui_def_model, self.area)
            self.ui_def_model.show()

    def compute_model(self):
        self.update_dm_slide_selected()
        if all([self.area.slides[sld].block_disp.disp_vec is None for sld in self.dm_slide_selected]):
            msg = QMessageBox()
            msg.setText("Please load necessary data")
            msg.setInformativeText("Data: DEM, Model, Displacement Data")
            msg.exec()
        else:
            if self.ui_comp_disp is None:
                self.ui_comp_disp = Ui_comp_disp()
            self.ui_compare_disp = UiConnectCompDisp(self, self.ui_comp_disp, self.area)
            self.ui_comp_disp.show()

    def help_docs(self):
        QDesktopServices.openUrl(QUrl("https://github.com/LeoLetellier/SLBL-FSA"))

    def help_file_format(self):
        QDesktopServices.openUrl(QUrl("https://github.com/LeoLetellier/SLBL-FSA/blob/master/SFA-stepbystep.pdf"))

    def help_about(self):
        QDesktopServices.openUrl(QUrl("https://wp.unil.ch/risk/"))

    def graph_setup(self):
        self.graph_layout = QVBoxLayout()
        self.area.graph.toolbar = NavigationToolbar(self.area.graph.canvas, self.other.frame)
        self.graph_layout.addWidget(self.area.graph.toolbar)
        self.graph_layout.addSpacing(4)
        self.graph_layout.addWidget(self.area.graph.canvas)
        self.other.frame.setLayout(self.graph_layout)
        self.graph_is_setup = True

    def graph_slbl(self):
        self.update_slide_selected()
        if self.area.slides[self.slide_selected].slbl.slbl_surf is None:
            msg = QMessageBox()
            msg.setText("This slide contains no SLBL surface!")
            msg.setInformativeText("Load or compute a surface before displaying.")
            msg.exec()
        else:
            self.area.graph.slbl_axis = self.update_axis(self.other.slbl_axis)
            self.area.graph.slbl_seq(self.area.slides[self.slide_selected].slbl.slbl_surf)
            self.area.graph.canvas.draw()
            if not self.graph_is_setup:
                self.graph_setup()
            self.area.graph.toolbar.show()
            print("DISPLAY: SLBL for " + self.slide_selected)

    def graph_disp_vec(self):
        self.update_dv_slide_selected()
        if self.area.slides[self.dv_slide_selected].block_disp.disp_vec is None:
            msg = QMessageBox()
            msg.setText("No displacement vectors!")
            msg.setInformativeText("Load or compute displacement vectors before displaying.")
            msg.exec()
        else:
            self.area.graph.vec_axis = self.update_axis(self.other.vec_axis)
            self.area.graph.disp_vec_seq(self.area.slides[self.dv_slide_selected])
            self.area.graph.canvas.draw()
            if not self.graph_is_setup:
                self.graph_setup()
            self.area.graph.toolbar.show()
            print("DISPLAY: Displacement vectors for " + self.dv_slide_selected)

    def graph_comp(self):
        if self.area.dc.disp_model is None or self.area.dc.disp_data is None:
            msg = QMessageBox()
            msg.setText("No Model and/or Displacement data!")
            msg.setInformativeText("Compute the model and load the displacement file before displaying.")
            msg.exec()
        else:
            self.area.dc.compute_diff_model_data()
            self.area.graph.comp_disp_seq()
            self.area.graph.canvas.draw()
            if not self.graph_is_setup:
                self.graph_setup()
            self.area.graph.toolbar.show()
            print("DISPLAY: data-model comparison")

    def graph_model(self):
        if self.area.model.disp_vec is None:
            msg = QMessageBox()
            msg.setText("No Model!")
            msg.setInformativeText("Compute the model before displaying.")
            msg.exec()
        else:
            self.area.graph.model_axis = self.update_axis(self.other.model_axis)
            self.update_dv_slide_selected()
            self.area.graph.disp_model_seq()
            self.area.graph.canvas.draw()
            if not self.graph_is_setup:
                self.graph_setup()
            self.area.graph.toolbar.show()
            print("DISPLAY: model")

    def graph_corr(self):
        if self.area.dc.disp_model is None or self.area.dc.disp_data is None:
            msg = QMessageBox()
            msg.setText("No Model and/or Displacement data!")
            msg.setInformativeText("Compute the model and load the displacement file before displaying.")
            msg.exec()
        else:
            self.area.dc.compute_diff_model_data()
            self.area.graph.corr_disp_seq()
            self.area.graph.canvas.draw()
            if not self.graph_is_setup:
                self.graph_setup()
            self.area.graph.toolbar.show()
            print("DISPLAY: correlation")

    def graph_clear(self):
        self.area.graph.fig_clear()
        self.area.graph.canvas.draw()
        self.area.graph.toolbar.hide()
        print("graph clear")


class UiConnectSLBL:
    def __init__(self, main_ui, other, area: LandslideArea):
        self.main_ui = main_ui
        other.box_left_lim.setMinimum(area.x[0])
        other.box_left_lim.setMaximum(area.x[-1])
        other.box_right_lim.setMinimum(area.x[0])
        other.box_right_lim.setMaximum(area.x[-1])
        self.area = area
        self.other = other
        other.btn_C_range.clicked.connect(self.compute_slbl_range)
        other.btn_C_combined.clicked.connect(self.compute_slbl_combined)
        other.btn_C_sub.clicked.connect(self.compute_slbl_sub)
        other.btn_C_point.clicked.connect(self.compute_slbl_point)

    def update_parameters(self):
        str_point = meter_2_idx(self.other.box_left_lim.value(), self.area.x)
        stp_point = meter_2_idx(self.other.box_right_lim.value(), self.area.x)
        self.area.slides[self.main_ui.slide_selected].start_pnt = str_point
        self.area.slides[self.main_ui.slide_selected].stop_pnt = stp_point
        self.area.slides[self.main_ui.slide_selected].slbl.c = self.other.box_tol.value()
        print("update parameters!", str_point, stp_point, self.area.slides[self.main_ui.slide_selected].slbl.c)

    def compute_slbl_range(self):
        self.main_ui.update_slide_selected()
        self.update_parameters()
        self.area.slides[self.main_ui.slide_selected].range_2_slbl()
        self.area.slides[self.main_ui.slide_selected].method = "RANGE"

    def compute_slbl_combined(self):
        self.main_ui.update_slide_selected()
        self.update_parameters()
        slide1 = "slide" + str(self.other.spinBox_comb1.value())
        slide2 = "slide" + str(self.other.spinBox_comb2.value())
        self.area.slides[self.main_ui.slide_selected].combine_slbl(slide1, slide2)
        self.area.slides[self.main_ui.slide_selected].method = "COMBINE"

    def compute_slbl_sub(self):
        self.main_ui.update_slide_selected()
        self.update_parameters()
        slide_master = "slide" + str(self.other.spinBox_master.value())
        self.area.slides[self.main_ui.slide_selected].sub_slbl(slide_master)
        self.area.slides[self.main_ui.slide_selected].method = "SUB"

    def compute_slbl_point(self):
        self.main_ui.update_slide_selected()
        self.update_parameters()
        self.area.slides[self.main_ui.slide_selected].point_slbl([self.other.x1.value(), self.other.z1.value()],
                                                                 [self.other.x2.value(), self.other.z2.value()])
        self.area.slides[self.main_ui.slide_selected].method = "POINT"


class UiConnectCompDisp:
    def __init__(self, main_ui, other, area: LandslideArea):
        self.main_ui = main_ui
        self.area = area
        self.other = other
        self.deformation = None
        other.btn_C_k.clicked.connect(self.compute_k)
        other.btn_C_rkms.clicked.connect(self.compute_rkms)
        # other.btn_C_k_rsq.clicked.connect(self.compute_rsq)

    def update_parameters(self):
        self.area.dc.alpha = self.other.doubleSpinBox_alpha.value()
        self.area.dc.theta = self.other.doubleSpinBox_theta.value()
        self.area.dc.delta = self.other.doubleSpinBox_delta.value()
        self.main_ui.update_dm_slide_selected()  # slide and ratio
        self.area.model.corresponding_slides = self.main_ui.dm_slide_selected
        self.area.dc.crop_disp_data()
        self.deformation = self.other.do_def.isChecked()

    def compute_k(self):
        self.update_parameters()
        if len(self.main_ui.dm_slide_selected) == 0:
            msg = QMessageBox()
            msg.setText("A slide contains no SLBL surface!")
            msg.setInformativeText("Load or compute a surface before displaying.")
            msg.exec()
        self.area.model.slides_2_model()
        if self.deformation:
            self.area.model.apply_table_a()
        self.area.dc.proj_disp(deformation=self.deformation)
        self.area.dc.method = 'USER'

    def compute_rkms(self):
        self.update_parameters()
        if self.area.dc.disp_data is None or len(self.main_ui.dm_slide_selected) == 0:
            msg = QMessageBox()
            msg.setText("No displacement file is loaded!")
            msg.setInformativeText("Load displacement file to ajust the model.")
            msg.exec()
        else:
            for k in self.main_ui.dm_slide_selected:
                print(k)
            self.area.dc.model_ajust_ratio_mean_square(deformation=self.deformation)
        self.area.dc.method = 'LEAST_SQUARE'

    # def compute_rsq(self):
    #     self.update_parameters()
    #     if self.area.dc.disp_data is None or len(self.main_ui.dm_slide_selected) == 0:
    #         msg = QMessageBox()
    #         msg.setText("No displacement file is loaded!")
    #         msg.setInformativeText("Load displacement file to ajust the model.")
    #         msg.exec()
    #     else:
    #         self.area.dc.model_ajust_ratio_rsq()

class UiConnectDefModel:
    def __init__(self, main_ui, other, area: LandslideArea):
        self.main_ui = main_ui
        self.area = area
        self.other = other
        other.load_a.clicked.connect(self.load_table)
        other.generate_table.clicked.connect(self.generate_table)
        other.correct_table.clicked.connect(self.correct_table)
        other.apply_a.clicked.connect(self.apply_table)
        other.save_a.clicked.connect(self.save_table)

    def update_parameters(self):
        if self.other.do_ap.isChecked():
            self.area.model.a_value = self.other.ap.value()
        elif self.other.do_an.isChecked():
            self.area.model.a_value = self.other.an.value()

    def load_table(self):
        self.main_ui.open_file(self.area.model.load_table_a)
        self.update_displayed_table()

    def save_table(self):
        self.main_ui.save_file(self.area.model.save_table_a)

    def generate_table(self):
        self.update_parameters()
        mdl = self.area.model
        mdl.generate_table()
        self.update_displayed_table()
        self.other.np.setText(str(floor(self.area.model.neutral_pnt)))
        self.other.vv.setText(str(self.area.model.volume_variation()))

    def update_displayed_table(self):
        mdl = self.area.model
        self.other.table.setRowCount(mdl.x_reg.shape[0]-2)
        self.other.table.setColumnCount(2)
        for pnt in range(mdl.x_reg.shape[0]-2):
            item = QTableWidgetItem(str(mdl.x_reg[1:-1][pnt]))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # non editable
            self.other.table.setItem(pnt, 0, item)
            self.other.table.setItem(pnt, 1, QTableWidgetItem(str(mdl.table_a[pnt])))
        self.other.table.resizeColumnsToContents()

    def correct_table(self):
        for pnt in range(self.area.model.x_reg.shape[0]-2):
            self.area.model.table_a[pnt] = float(self.other.table.item(pnt, 1).text())
        self.other.vv.setText(str(self.area.model.volume_variation()))
        print("DONE: correct table")

    def apply_table(self):
        self.area.model.apply_table_a()
