"""
Python file containing the features and functions relative to the processing and handling variables
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import gridspec
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from scipy.stats import linregress
import scipy.interpolate as itp
from scipy.optimize import least_squares
from math import atan
from matplotlib.colors import LinearSegmentedColormap

from SlblToolKit import *


class LandslideArea:
    """
    Master class containing data about the landslide for SLBL and displacement features
    """

    def __init__(self):
        self.x = None  # Input DEM
        self.z = None  # Input DEM
        self.dem_path = None
        self.slides = {}
        self.model = DispModel(self)
        self.dc = DispComp(self)
        self.graph = Graph(self)

    def access_slide(self, name: str):
        """
        Create or access a slide given its name
        :param name: name of the slide
        :return: slide instance
        """
        if name not in self.slides.keys():
            self.slides[name] = Slide(self)
        return self.slides[name]

    def delete_slide(self, name: str):
        """
        Delete a slide given its name
        :param name: name of the slide
        :return: None
        """
        if name in self.slides.keys:
            del self.slides[name]

    def reset_model(self):
        """
        Reset the model to default values
        :return: None
        """
        self.model = DispModel(self)

    def load_dem(self, data_path):
        """
        | x | z |
        :param data_path: path where data has to be retrieved
        :return: None
        """
        df = read_data_file(data_path, 0, 1)
        self.x = np.asarray(df.iloc[:, 0])
        self.z = np.asarray(df.iloc[:, 1])
        print("LOAD: DEM")


class DispModel:
    """
    Class centralizing shared features
    """

    def __init__(self, area):
        self.area = area
        self.start_pnt = None
        self.stop_pnt = None
        self.n_vec = None  # nb of final vectors to be computed
        self.x_reg = None
        self.z_reg = None
        self.slope_direction = None
        self.disp_vec = None
        self.disp_with_def = None
        self.corresponding_slides = []

        self.a_value = None
        self.neutral_pnt = None
        self.volume_var = None
        self.table_a = None

    def load_table_a(self, data_path):
        """
        Load the coefficient table for model with deformation
        :param data_path: path where data has to be retrieved
        :return: None
        """
        df = read_data_file(data_path, 1)
        self.table_a = np.asarray(df)
        print("LOAD: table")

    def save_table_a(self, data_path):
        """
        Save the coefficient table for model with deformation
        :param data_path: path where data has to be saved
        :return: None
        """
        data_2_save = {'x_reg': self.x_reg[1:-1], 'coefficient': self.table_a}
        save_csv_file(data_path, data_2_save)
        print("SAVE: table")

    def generate_table(self):
        """
        Generate a table for model with deformation using an extremity coefficient
        :return: None
        """
        x_reg = np.copy(self.x_reg[1:-1])
        self.table_a = np.zeros(shape=x_reg.shape)
        cst = np.sum(x_reg * self.disp_vec[:, 1]) / np.sum(self.disp_vec[:, 1])
        if self.slope_direction * self.a_value > 0:
            x_loc = x_reg[0]
        else:
            x_loc = x_reg[-1]
        if x_loc - cst != 0:
            c = self.a_value / (x_loc - cst)
            self.neutral_pnt = cst
            for pnt in range(self.table_a.shape[0]):
                if x_loc != x_reg[pnt]:
                    self.table_a[pnt] = c * (x_reg[pnt] - cst)
                else:
                    self.table_a[pnt] = self.a_value
            print("DONE: generate table")
        else:
            print("ERROR: division by zero encountered. a value must be different from 0")

    def volume_variation(self) -> float:
        """
        Compute the difference of volume (based on displacement variation)
        :return: volume variation
        """
        return np.sum(self.table_a * self.disp_vec[:, 1])

    def apply_table_a(self):
        """
        Apply the table of coefficient for model with deformation on previous model
        :return: None
        """
        self.disp_with_def = np.copy(self.disp_vec)
        if np.sum(self.disp_vec[:, 1]) < 0:
            self.disp_with_def[:, 1] -= self.disp_with_def[:, 1] * self.table_a
        else:
            self.disp_with_def[:, 1] += self.disp_with_def[:, 1] * self.table_a
        self.volume_var = self.volume_variation()
        print("DONE: apply table")

    def compute_interp(self):
        """
        Compute the interpolation of the topography on the points of vectors
        :return: None
        """
        # We need k-1 and k+1 so the two extremities outside range
        # retstep allows the return of step
        self.x_reg, step = np.linspace(self.area.x[self.start_pnt], self.area.x[self.stop_pnt], self.n_vec,
                                       retstep=True, endpoint=True)
        self.x_reg = np.append(np.append(self.x_reg[0] - step, self.x_reg), self.x_reg[-1] + step)
        f_interp_topo = itp.interp1d(self.area.x, self.area.z, kind='linear')
        self.z_reg = f_interp_topo(self.x_reg)

    def slides_2_model(self):
        """
        Combine the selected slides considering their ratios and global elongation profile
        :return: None
        """
        self.disp_vec = np.zeros([self.n_vec, 2])
        for sld in self.corresponding_slides:
            slide = self.area.slides[sld]
            xz_vec = np.column_stack((slide.block_disp.vec_x_reg, slide.block_disp.vec_z_reg))
            xz_vec = np.nan_to_num(xz_vec)
            self.disp_vec += slide.disp_ratio * xz_vec

    def save(self, data_path):
        """
        Save the model in csv file
        :param data_path: path where data has to be stored
        :return: None
        """
        if self.area.dc.diff_model_data is not None:
            zero = np.array([0.])
            data_2_save_1 = {'xdata': self.area.dc.x_data, 'disp_model': self.area.dc.disp_model_on_data,
                             'disp_data': self.area.dc.disp_data, 'disp_diff': self.area.dc.diff_model_data}
            data_2_save_2 = {'x_reg': self.area.model.x_reg, 'z_reg': self.area.model.z_reg,
                             'disp_vec_x': np.append(zero, np.append(self.area.model.disp_vec[:, 0], zero)),
                             'disp_vec_z': np.append(zero, np.append(self.area.model.disp_vec[:, 1], zero))}
            save_csv_file(data_path, pd.concat([pd.DataFrame(data_2_save_1), pd.DataFrame(data_2_save_2)], axis=1))
            print("SAVE: Model")
        else:
            print("ERROR: no data to be saved")

    def reset_model(self):
        """
        Reset the model to default values
        :return: None
        """
        self.n_vec = None
        self.x_reg = None
        self.z_reg = None
        self.elong_reg = None
        self.elongation_profile = None
        self.disp_vec = None


class Slide:
    """
    Class centralizing specific features for one particular SLBL
    """

    def __init__(self, area):
        self.area = area
        self.slide_path = None
        self.method = None
        self.start_pnt = None  # Index
        self.stop_pnt = None  # Index
        self.disp_ratio = 1  # Ratio of relative impact on total movement
        self.slbl = Slbl(self)
        self.block_disp = BlockDisp(self)
        self.lin_value_left = None
        self.lin_value_right = None
        self.comb1 = None
        self.comb2 = None
        self.master = None
        self.points = None

    def range_2_slbl(self):
        """
        Compute the SLBL using the range on cross-section
        :return: None
        """
        self.slbl.compute_slbl_matrix()
        print("COMPUTED: range_to_slbl")

    def combine_slbl(self, slide1: str, slide2: str):
        """
        Compute the SLBL by combining two SLBLs (minimum wise)
        :param slide1: name of slide
        :param slide2: name of slide
        :return: None
        """
        self.comb1 = slide1
        self.comb2 = slide2
        self.slbl.slbl_surf = np.minimum(self.area.slides[slide1].slbl.slbl_surf,
                                         self.area.slides[slide2].slbl.slbl_surf)
        print("COMPUTED: combine_slbl")

    def sub_slbl(self, slide_master: str):
        """
        Compute the SLBL by constraining a range-SLBL with an existing feature (SLBL, fault, ...)
        :param slide_master: name of slide
        :return:None
        """
        self.master = slide_master
        self.slbl.compute_slbl_matrix()
        self.slbl.slbl_surf = np.maximum(self.slbl.slbl_surf, self.area.slides[slide_master].slbl.slbl_surf)
        print("COMPUTED: sub_slbl")

    def point_slbl(self, point1: list, point2: list):
        """
        Compute the SLBL using the coordinates of the two extremities. If a point is in range, the z value will be taken
        from DEM.
        :param point1: coordinates of the left extremity
        :param point2: coordinates of the right extremity
        :return: None
        """
        print("point1", point1)
        print("point2", point2)
        self.points = [point1, point2]
        x = np.linspace(point1[0], point2[0], 200)
        z = np.zeros(shape=x.shape)
        z[0] = point1[1]
        z[-1] = point2[1]
        self.slbl.compute_slbl_matrix_custom(z)
        itp_surf = itp.interp1d(x, self.slbl.slbl_surf)
        self.slbl.slbl_surf = np.copy(self.area.z)
        for pnt in range(self.slbl.slbl_surf.shape[0]):
            if point1[0] < self.area.x[pnt] < point2[0]:
                self.slbl.slbl_surf[pnt] = itp_surf(self.area.x[pnt])
        self.slbl.slbl_surf = np.minimum(self.slbl.slbl_surf, self.area.z)
        print("DONE: point_slbl")

    def reset_bd(self):
        """
        Reset the block displacement to default values
        :return: None
        """
        self.block_disp.reg_start = None
        self.block_disp.reg_stop = None
        self.block_disp.slbl_reg = None
        self.block_disp.disp_vec = None
        self.block_disp.angles = None
        self.block_disp.block_lines = None
        self.block_disp.vec_x_reg = None
        self.block_disp.vec_z_reg = None

    def linear_profile_on_bd(self):
        """
        Add a linear profile of elongation (norm) along the displacement of one slide
        :return: None
        """
        n = self.block_disp.reg_stop - self.block_disp.reg_start + 1
        indices = np.arange(n)
        factor = indices / n * self.lin_value_right + (n - indices) / n * self.lin_value_left
        factor = np.append(np.zeros(self.block_disp.reg_start - 1),
                           np.append(factor, np.zeros(self.area.model.n_vec - self.block_disp.reg_stop)))
        print("shape disp vec", self.block_disp.disp_vec.shape)
        print("shape factor", factor.shape)
        self.block_disp.disp_vec *= factor[:, np.newaxis]
        self.block_disp.interp_vec()


class Slbl:
    """
    Class containing all functions and variables relative to SLBL processing
    """

    def __init__(self, slide: Slide):
        self.slide = slide
        self.slbl_surf = None

        self.x = None
        self.c = None
        self.dim = None
        self.mc = None
        self.main_diag = None
        self.subs_diag = None

    def load(self, data_path: str):
        """
        | x | slbl |
        :param data_path: path where data has to be retrieved
        """
        df = read_data_file(data_path, 1)
        self.slbl_surf = np.asarray(df)
        tol = 0.5
        for xi in range(self.slbl_surf.shape[0]):
            if self.slide.start_pnt is None and abs(self.slide.area.z[xi] - self.slbl_surf[xi]) > tol:
                self.slide.start_pnt = xi - 1
            if self.slide.start_pnt is not None and self.slide.stop_pnt is None and abs(
                    self.slide.area.z[xi] - self.slbl_surf[xi]) < tol:
                self.slide.stop_pnt = xi
        print("LOAD: SLBL")

    def save_computed(self, data_path):
        """
        | x | slbl |
        :param data_path: path where data has to be stored
        """
        if self.slbl_surf is not None:
            data_2_save = {"x": self.slide.area.x, "slbl": self.slbl_surf}
            save_csv_file(data_path, data_2_save)
            print("SAVE: SLBL")
        else:
            print("ERROR: no SLBL data to save")

    def compute_slbl_matrix(self):
        """
        Compute the solution to the matrix linear equation using the tridiagonal matrix algorithm
        """
        strt_pnt = self.slide.start_pnt
        stp_pnt = self.slide.stop_pnt
        self.dim = stp_pnt - strt_pnt - 1
        self.x = np.empty(self.dim)
        self.main_diag = np.ones(self.dim)
        self.subs_diag = -0.5 * np.ones(self.dim - 1)
        self.mc = - self.c * np.ones(self.dim)
        self.mc[0] += self.slide.area.z[strt_pnt] / 2
        self.mc[-1] += self.slide.area.z[stp_pnt] / 2
        b = np.copy(self.main_diag)
        d = np.copy(self.mc)
        for i in range(1, self.dim):
            w = self.subs_diag[i - 1] / b[i - 1]
            b[i] = b[i] - w * self.subs_diag[i - 1]
            d[i] = d[i] - w * d[i - 1]
        self.x[self.dim - 1] = d[self.dim - 1] / b[self.dim - 1]
        for i in range(self.dim - 2, -1, -1):
            self.x[i] = (d[i] - self.subs_diag[i] * self.x[i + 1]) / b[i]
        self.slbl_surf = np.copy(self.slide.area.z)
        self.slbl_surf[strt_pnt + 1:stp_pnt] = self.x

    def compute_slbl_matrix_custom(self, z):
        """
        Compute the slbl with custom matrix, used for point slbl
        :param z: pattern for output and extremities
        :return: None
        """
        dim = z.shape[0] - 2
        self.x = np.empty(dim)
        self.main_diag = np.ones(dim)
        self.subs_diag = -0.5 * np.ones(dim - 1)
        self.mc = - self.c * np.ones(dim)
        self.mc[0] += z[0] / 2
        self.mc[-1] += z[-1] / 2
        b = np.copy(self.main_diag)
        d = np.copy(self.mc)
        for i in range(1, dim):
            w = self.subs_diag[i - 1] / b[i - 1]
            b[i] = b[i] - w * self.subs_diag[i - 1]
            d[i] = d[i] - w * d[i - 1]
        self.x[dim - 1] = d[dim - 1] / b[dim - 1]
        for i in range(dim - 2, -1, -1):
            self.x[i] = (d[i] - self.subs_diag[i] * self.x[i + 1]) / b[i]
        self.slbl_surf = np.copy(z)
        self.slbl_surf[1:dim + 1] = self.x


class BlockDisp:
    """
    Class containing all functions and variables relative to block displacement and vectors
    """

    def __init__(self, slide: Slide):
        self.slide = slide
        self.delta_x = 400
        self.reg_start = None
        self.reg_stop = None
        self.slbl_reg = None
        self.disp_vec = None
        self.angles = None
        self.block_lines = None
        self.vec_x_reg = None
        self.vec_z_reg = None

    def prep_data(self):
        """
        Compute interpolate data for block computation
        """
        if self.slide.area.model.x_reg is None:
            self.slide.area.model.compute_interp()
        x_reg = self.slide.area.model.x_reg
        n_vec = self.slide.area.model.n_vec
        self.reg_start = meter_2_idx(self.slide.area.x[self.slide.start_pnt], x_reg)
        self.reg_stop = meter_2_idx_before(self.slide.area.x[self.slide.stop_pnt], x_reg)
        self.disp_vec = np.zeros([n_vec, 2])
        f_interp_slbl = itp.interp1d(self.slide.area.x, self.slide.slbl.slbl_surf, kind='linear')
        self.slbl_reg = f_interp_slbl(x_reg)
        self.block_lines = np.empty([n_vec, 4])
        self.angles = np.empty(n_vec)
        self.block_lines[:, :] = np.nan
        self.angles[:] = np.nan

    def compute_disp_vectors(self):
        """
        Compute the displacement vectors and blocks associated to the reference displacement values
        """
        self.prep_data()
        x_reg = self.slide.area.model.x_reg
        z_reg = self.slide.area.model.z_reg
        for k in range(self.reg_start, self.reg_stop + 1):
            m = (self.slbl_reg[k + 1] - self.slbl_reg[k - 1]) / (x_reg[k + 1] - x_reg[k - 1])
            h = self.slbl_reg[k] + (x_reg[k]) / m
            self.angles[k - 1] = atan(m) * 180 / pi
            if k == self.reg_start or k == self.reg_stop:
                # the two extremities of the landslide are fixed points
                # We can't use intersection_on_topo because the intersection could be outside of range
                # and thus return just None
                intersection = [x_reg[k], z_reg[k]]
            else:
                xx = [x_reg[k] - self.delta_x, x_reg[k] + self.delta_x]
                zz = [h - xx[0] / m, h - xx[1] / m]
                intersection = intersection_on_topo(x_reg, z_reg, xx, zz)
            self.disp_vec[k - 1, :] = np.array(
                [cos(self.angles[k - 1] * pi / 180),
                 sin(self.angles[k - 1] * pi / 180)]) * -self.slide.area.model.slope_direction
            if intersection is not None:
                self.block_lines[k - 1] = np.array([x_reg[k], intersection[0], self.slbl_reg[k], intersection[1]])
        self.interp_vec()
        print("DONE: disp vec compute")

    def interp_vec(self):
        """
        Compute interpolate vectors for model computation
        :return: None
        """
        x_reg = self.slide.area.model.x_reg
        f_interp_vec_x = itp.interp1d(self.block_lines[:, 1], self.disp_vec[:, 0], fill_value='extrapolate')
        f_interp_vec_z = itp.interp1d(self.block_lines[:, 1], self.disp_vec[:, 1], fill_value='extrapolate')
        # Behaviour from fill_value='extrapolate' is uncertain, so outside values set manually to 0
        # print(self.block_lines[:, 1], self.disp_vec[:, 0], self.disp_vec[:, 1])
        self.vec_x_reg = f_interp_vec_x(x_reg[1:-1])
        self.vec_x_reg[:self.reg_start - 1] = 0
        self.vec_x_reg[self.reg_stop:] = 0
        self.vec_z_reg = f_interp_vec_z(x_reg[1:-1])
        self.vec_z_reg[:self.reg_start - 1] = 0
        self.vec_z_reg[self.reg_stop:] = 0

    def test_crossing(self) -> bool:
        """
        Test if some projection of vectors on the topography are crossing together (lines displayed on graph)
        :return: True if no crossing, False if at least one crossing
        """
        for k in range(self.block_lines.shape[0] - 1):
            bl = self.block_lines[k]
            bl2 = self.block_lines[k + 1]
            if get_intersection_point([bl[0], bl[1]], [bl[2], bl[3]], [bl2[0], bl2[1]], [bl2[2], bl2[3]]) is not None:
                print("WARNING: test crossing in block lines")
                return False
        print("DONE: test crossing in block lines passed")
        return True

    def save(self, data_path):
        """
        | x_reg | x1 | z1 | x2 | z2 | vx | vz | angles |
        :param data_path: path where data has to be stored
        :return: None
        """
        if self.disp_vec is not None:
            zero = np.array([0.])
            data_2_save = {'x_reg': self.slide.area.model.x_reg,
                           'x1': np.append(zero, np.append(self.block_lines[:, 0], zero)),
                           'z1': np.append(zero, np.append(self.block_lines[:, 2], zero)),
                           'x2': np.append(zero, np.append(self.block_lines[:, 1], zero)),
                           'z2': np.append(zero, np.append(self.block_lines[:, 3], zero)),
                           'vx': np.append(zero, np.append(self.disp_vec[:, 0], zero)),
                           'vy': np.append(zero, np.append(self.disp_vec[:, 1], zero)),
                           'angles': np.append(zero, np.append(self.angles, zero))}
            save_csv_file(data_path, data_2_save)
            print("SAVE: Block displacement")
        else:
            print("ERROR: no block disp data to save")

    def load(self, data_path):
        """
        Load block disp data
        :param data_path: path where data has to be retrieved
        :return: None
        """
        df = read_data_file(data_path, 0, 7)
        if self.slide.area.model.x_reg is None:
            self.slide.area.model.x_reg = np.asarray(df.iloc[:, 0])
        if np.array_equal(self.slide.area.model.x_reg, np.asarray(df.iloc[:, 0])):
            self.block_lines = np.column_stack((np.asarray(df.iloc[1:-1, 1]), np.asarray(df.iloc[1:-1, 2]),
                                                np.asarray(df.iloc[1:-1, 3]), np.asarray(df.iloc[1:-1, 4])))
            self.disp_vec = np.column_stack((np.asarray(df.iloc[1:-1, 5]), np.asarray(df.iloc[1:-1, 6])))
            self.angles = np.asarray(df.iloc[1:-1, 7])
            self.slide.area.model.n_vec = self.angles.shape[0]
            self.interp_vec()
            print("LOAD: displacement data")
        else:
            print("ERROR: data not matching")


class DispComp:
    """
    Class containing all functions and variables relative to displacement comparison
    """

    def __init__(self, area: LandslideArea):
        self.area = area
        self.disp_path = None
        self.alpha = None
        self.theta = None
        self.delta = None
        self.method = 'USER'
        self.deformation = False

        self.disp_model = None
        self.disp_model_on_data = None  # disp_model interpolated on x_data
        self.x_data = None
        self.full_x = None
        self.full_data = None
        self.disp_data = None
        self.mobile_mean_data = None
        self.diff_model_data = None
        self.rmse = None
        self.r2 = None
        self.regression = None

    def load(self, data_path):
        """
        | x | ext_disp |
        :param data_path: path where data has to be retrieved
        :return: None
        """
        df = read_data_file(data_path, 0, 1)
        self.full_x = np.asarray(df.iloc[:, 0])
        self.full_data = np.asarray(df.iloc[:, 1])

        self.crop_disp_data()
        if self.disp_model is not None:
            self.disp_model_2_data()
        print("LOAD: displacement data")

    def crop_disp_data(self):
        """
        Crop the displacement data to the x range of the model
        :return: None
        """
        self.disp_data = np.copy(self.full_data)[self.full_x > self.area.x[self.area.model.start_pnt]]
        self.x_data = np.copy(self.full_x)[self.full_x > self.area.x[self.area.model.start_pnt]]
        self.disp_data = self.disp_data[self.x_data < self.area.x[self.area.model.stop_pnt]]
        self.x_data = self.x_data[self.x_data < self.area.x[self.area.model.stop_pnt]]

        self.mobile_mean_data = interpolation_moving_mean(self.x_data, self.disp_data, self.area.model.x_reg)

    def proj_disp(self, deformation=False):
        """
        Compute the normal vectors of the LOS and the cross-section, and compute the projected displacement
        :return: None
        """
        self.disp_model = np.zeros(self.area.model.n_vec)
        vec_los = normal_vector_los(self.theta, self.delta)
        if deformation is False:
            vec = self.area.model.disp_vec
        else:
            vec = self.area.model.disp_with_def
        for pnt in range(self.area.model.n_vec):
            if vec[pnt, 0] == 0:
                self.disp_model[pnt] = 0
            else:
                # Adjust azimuth
                l_dir = 1 if vec[pnt, 0] >= 0 else -1
                # Adjust incidence
                angle = atan(vec[pnt, 1] / abs(vec[pnt, 0])) * 180 / pi if vec[pnt, 0] != 0 else 0
                incidence = abs(angle) + pi / 2 if angle > 0 else pi / 2 - abs(angle)
                # Compute inner product of unit vectors = projection
                vec_local_section = normal_vector_los(incidence, l_dir * self.alpha)
                # Weight by the amplitude of displacement
                self.disp_model[pnt] = np.dot(vec_local_section, vec_los) * np.linalg.norm(vec[pnt])
        if self.x_data is not None:
            self.disp_model_2_data()

    def disp_model_2_data(self):
        """
        Interpolate the model on data point
        :return: None
        """
        f_interp_disp_model = itp.interp1d(self.area.model.x_reg[1:-1], self.disp_model, kind='linear')
        self.disp_model_on_data = -f_interp_disp_model(self.x_data)  # matching los convention

    def compute_rmse(self):
        """
        Compute the RMSE between data and model
        :return: None
        """
        self.rmse = np.sum((self.disp_model_on_data - self.disp_data) ** 2) / self.disp_data.shape[0]
        print("RMSE:", self.rmse)

    def compute_r_square(self):
        """
        Compute the r squared between data and model
        :return: None
        """
        self.regression = linregress(self.disp_data, self.disp_model_on_data)
        self.r2 = self.regression[2] ** 2
        print("LINEAR REGRESSION:", self.regression)

    def compute_diff_model_data(self):
        """
        Compute the difference between data and model
        => DATA - MODEL
        :return: None
        """
        self.diff_model_data = self.disp_data - self.disp_model_on_data
        self.compute_rmse()
        self.compute_r_square()
        print("DONE: compute difference between data and model")

    def model_ajust_ratio_mean_square(self, deformation=False):
        """
        Optimize ratios by least square on RMSE
        :return: None
        """
        self.deformation = deformation
        x0 = [1.] * len(self.area.model.corresponding_slides)
        print("x0 ", x0)
        result = least_squares(self.ratio_function, x0)
        ratios = result.x
        print("LEAST SQUARE RESULTS:", result)
        print("RATIOS:", ratios)
        for sld in range(len(self.area.model.corresponding_slides)):
            self.area.slides[self.area.model.corresponding_slides[sld]].disp_ratio = abs(ratios[sld])
        self.area.model.slides_2_model()
        if deformation:
            self.area.model.apply_table_a()
        self.proj_disp(deformation)
        self.deformation = False
        print("DONE: ajust k and ratios with least square")

    # def model_ajust_ratio_rsq(self):
    #     """
    #     Optimize ratios by least square on r²
    #     :return: None
    #     """
    #     result = least_squares(self.rsq_function, [1.] * len(self.area.model.corresponding_slides))
    #     ratios = result.x
    #     print("LEAST SQUARE RESULTS:", result)
    #     print("RATIOS:", ratios)
    #     for sld in range(len(self.area.model.corresponding_slides)):
    #         self.area.slides[self.area.model.corresponding_slides[sld]].disp_ratio = abs(ratios[sld])
    #     self.area.model.slides_2_model()
    #     self.proj_disp()
    #     print("DONE: ajust k and ratios with least square")

    def ratio_function(self, r):
        """
        Function to minimize by least square using RMSE
        :param r: array of parameters to influence
        :return: None
        """
        for sld in range(len(self.area.model.corresponding_slides)):
            self.area.slides[self.area.model.corresponding_slides[sld]].disp_ratio = abs(r[sld])
        self.area.model.slides_2_model()
        if self.deformation:
            self.area.model.apply_table_a()
        self.proj_disp(deformation=self.deformation)
        return np.sum((self.disp_model_on_data - self.disp_data) ** 2)

    # def rsq_function(self, r):
    #     """
    #     Function to minimize by least square using r²
    #     :param r: array of parameters to influence
    #     :return: None
    #     """
    #     for sld in range(len(self.area.model.corresponding_slides)):
    #         self.area.slides[self.area.model.corresponding_slides[sld]].disp_ratio = abs(r[sld])
    #     self.area.model.slides_2_model()
    #     self.proj_disp()
    #     regression = linregress(self.disp_data, self.disp_model_on_data)
    #     return 1 - regression[2] ** 2


class Graph:
    """
    Class containing functions relative to graph display
    """

    def __init__(self, area: LandslideArea):
        self.area = area
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = None
        self.is_clear = True
        self.slbl_axis = 'auto'
        self.vec_axis = 'auto'
        self.model_axis = 'auto'
        self.scale = None

    def plot_dem(self, ax):
        """
        Plot the topography onto the given axis into a 2D cross-section
        :param ax: matplotlib axis
        """
        ax.plot(self.area.x, self.area.z, '-k', label='DEM')

    def plot_slbl(self, ax, slbl, ratio=None):
        """
        Plot the failure surface onto the given axis into a 2D cross-section
        :param ax: matplotlib axis
        :param slbl: slbl data matching x_axis
        """
        if ratio is not None:
            ax.plot(self.area.x, slbl, "-.", label='SLBL (r=' + f"{ratio:.6f}" + ')')
        else:
            ax.plot(self.area.x, slbl, "-.", label='SLBL')

    def plot_blocks_vec(self, ax, slide):
        """
        Plot the division blocks along with the displacement vectors into a 2D cross-section
        :param ax: matplotlib axis
        """
        bl = np.copy(slide.block_disp.block_lines)
        dv = np.copy(slide.block_disp.disp_vec)
        index = ~np.isnan(bl[:, 0])
        bl = bl[index]
        dv = dv[index]
        ax.plot([bl[:, 0], bl[:, 1]], [bl[:, 2], bl[:, 3]], '-C7')
        q = ax.quiver(bl[:, 1], bl[:, 3], dv[:, 0], dv[:, 1], angles='xy', scale_units='xy', width=0.0016, headwidth=2)

    def plot_disp(self, ax):
        """
        Plot the displacement into a 2D cross-section
        :param ax: matplotlib axis
        """
        # for k in range(np.shape(self.area.dc.x_data)[0]):
        ax.plot([self.area.dc.x_data] * 2, [self.area.dc.disp_model_on_data, self.area.dc.disp_data], "-k")
        ax.scatter(self.area.dc.x_data, self.area.dc.disp_model_on_data, color='red', label='model')
        ax.scatter(self.area.dc.x_data, self.area.dc.disp_data, color='black', label='data')

    def plot_diff(self, ax):
        """
        Plot the difference between model and data
        :param ax: matplotlib axis
        :return: None
        """
        ax.plot(self.area.dc.x_data, self.area.dc.diff_model_data, "-r", label='data-model difference')
        ax.fill_between(self.area.dc.x_data, self.area.dc.diff_model_data, color='red', alpha=0.5)

    def plot_model(self, ax):
        """
        Plot the model (combines SLBLs)
        :param ax: matplotlib axis
        :return: None
        """
        dv = self.area.model.disp_vec
        x = np.copy(self.area.model.x_reg[1:-1])
        y = np.copy(self.area.model.z_reg[1:-1])
        u = np.copy(dv[:, 0])
        v = np.copy(dv[:, 1])
        nb = "{:e}".format(np.max(np.sqrt(u ** 2 + v ** 2)))
        amp = float(nb.split("e")[0][0]) * 10 ** int(nb.split("e")[1])
        self.scale = amp
        if self.area.model.slope_direction > 0:
            x = np.append(x, np.min(self.area.x))
            y = np.append(y, np.max(self.area.z))
            u = np.append(u, amp)
            v = np.append(v, 0)
        else:
            x = np.append(x, np.max(self.area.x))
            y = np.append(y, np.max(self.area.z))
            u = np.append(u, -amp)
            v = np.append(v, 0)
        q = ax.quiver(x, y, u, v, angles='xy', scale_units='xy', width=0.0016, headwidth=2,
                      label='scale=' + str(amp) + "cm")

    def add_los_on_model(self, ax):
        """
        Add the los view in cross-section with the norm at the displacement at the point of data
        :param ax: ax to display on
        :return: None
        """
        vec_los = normal_vector_los(self.area.dc.theta, self.area.dc.delta)
        vec_section = normal_vector_los(90, self.area.dc.alpha)
        vec_los_in_section = [np.dot(vec_los, vec_section), vec_los[2]]
        itp_topo = itp.interp1d(self.area.x, self.area.z)
        x = np.copy(self.area.dc.x_data)
        y = np.copy(itp_topo(self.area.dc.x_data))
        u = np.copy(-vec_los_in_section[0] * self.area.dc.disp_data)
        v = np.copy(-vec_los_in_section[1] * self.area.dc.disp_data)
        # nb = "{:e}".format(np.max(np.sqrt(u ** 2 + v ** 2)))
        # amp = float(nb.split("e")[0][0]) * 10 ** int(nb.split("e")[1])
        amp = self.scale
        if self.area.model.slope_direction > 0:
            x = np.append(x, np.min(self.area.x))
            y = np.append(y, np.min(self.area.z) + (np.max(self.area.z) - np.min(self.area.z)) * 7 / 8)
            u = np.append(u, amp)
            v = np.append(v, 0)
        else:
            x = np.append(x, np.max(self.area.x))
            y = np.append(y, np.min(self.area.z) + (np.max(self.area.z) - np.min(self.area.z)) * 7 / 8)
            u = np.append(u, -amp)
            v = np.append(v, 0)
        q = ax.quiver(x, y, u, v, angles='xy', scale_units='xy', width=0.0012, headwidth=2, color='red',
                      label='scale=' + str(amp) + "cm")
        # Be careful of the sign of the displacement

    def plot_model_and_los(self, ax):
        dv = self.area.model.disp_vec
        x_model = np.copy(self.area.model.x_reg[1:-1])
        y_model = np.copy(self.area.model.z_reg[1:-1])
        u_model = np.copy(dv[:, 0])
        v_model = np.copy(dv[:, 1])

        vec_los = normal_vector_los(self.area.dc.theta, self.area.dc.delta)
        vec_section = normal_vector_los(90, self.area.dc.alpha)
        vec_los_in_section = np.array([np.sign(np.dot(vec_los, vec_section)) * np.sqrt(1 - np.square(vec_los[2])), vec_los[2]])
        vec_los_in_section *= np.abs(np.dot(vec_los, vec_section))
        itp_topo = itp.interp1d(self.area.x, self.area.z)
        x_los = np.copy(self.area.dc.x_data)
        y_los = np.copy(itp_topo(self.area.dc.x_data))
        u_los = np.copy(-vec_los_in_section[0] * self.area.dc.disp_data)
        v_los = np.copy(-vec_los_in_section[1] * self.area.dc.disp_data)

        x = np.append(x_model, x_los)
        y = np.append(y_model, y_los)
        u = np.append(u_model, u_los)
        v = np.append(v_model, v_los)

        nb = "{:e}".format(np.max(np.sqrt(u ** 2 + v ** 2)))
        amp = float(nb.split("e")[0][0]) * 10 ** int(nb.split("e")[1])
        if self.area.model.slope_direction > 0:
            x = np.append(x, np.min(self.area.x))
            y = np.append(y, np.max(self.area.z))
            u = np.append(u, amp)
            v = np.append(v, 0)
        else:
            x = np.append(x, np.max(self.area.x))
            y = np.append(y, np.max(self.area.z))
            u = np.append(u, -amp)
            v = np.append(v, 0)
        colors = np.append(np.full(x_model.shape, 0), np.append(np.full(x_los.shape, 0.5), np.array(1)))
        colormap = LinearSegmentedColormap.from_list('quiver_color', ["#000000", "#FF0000", "#00c9ff"])
        q = ax.quiver(x, y, u, v, color=colormap(colors), angles='xy', scale_units='xy', width=0.0016, headwidth=2,
                      label='scale=' + str(amp) + "cm")

    def plot_corr(self, ax):
        """
        Draw the correlation between data values and model predicted values and comparing to 1:1 line to discuss r²
        :param ax: axis to display on
        :return: None
        """
        ax.scatter(self.area.dc.disp_data, self.area.dc.disp_model_on_data, marker='o', color='blue')
        min = np.min([np.min(self.area.dc.disp_data), np.min(self.area.dc.disp_model_on_data)])
        max = np.max([np.max(self.area.dc.disp_data), np.max(self.area.dc.disp_model_on_data)])
        line = np.linspace(min, max, 100)
        ax.plot(line, line, color='gray', label='1:1 line')
        a = self.area.dc.regression[0]
        b = self.area.dc.regression[1]
        ax.plot(line, a * line + b, color='black', label='regression line')
        ax.axis('equal')

    def plot_moving_mean(self, ax):
        """
        Draw the moving mean interpolation of the displacement data
        :param ax: axis to display on
        :return: None
        """
        x = np.copy(self.area.model.x_reg)[~np.isnan(self.area.dc.mobile_mean_data)]
        y = np.copy(self.area.dc.mobile_mean_data)[~np.isnan(self.area.dc.mobile_mean_data)]
        ax.plot(x, y, "-o")

    def fig_clear(self):
        """
        Clear the figure display
        """
        if not self.is_clear:
            self.fig.clf()
            self.is_clear = True

    def slbl_seq(self, slbl):
        """
        All functions to call to diplay the SLBL graph
        :param slbl: chosen SLBL (input or computed)
        """
        self.fig_clear()
        self.is_clear = False
        self.ax = self.fig.add_subplot(111)
        self.plot_slbl(self.ax, slbl)
        self.plot_dem(self.ax)
        self.fig.legend()
        self.ax.set_xlabel("Distance (m)")
        self.ax.set_ylabel("Elevation (m)")
        self.ax.axis(self.slbl_axis)
        self.fig.subplots_adjust(hspace=0.3)
        self.fig.legend()
        self.fig.suptitle("SLBL overview")

    def disp_vec_seq(self, slide):
        """
        All functions to call to display the block displacement graphs
        :param slide: chosen slide
        """
        self.fig_clear()
        self.is_clear = False
        self.ax = self.fig.add_subplot(111)
        self.plot_blocks_vec(self.ax, slide)
        self.plot_slbl(self.ax, slide.slbl.slbl_surf)
        self.plot_dem(self.ax)
        self.fig.legend()
        self.ax.set_xlabel("Distance (m)")
        self.ax.set_ylabel("Elevation (m)")
        self.ax.axis(self.vec_axis)
        self.fig.subplots_adjust(hspace=0.3)
        self.fig.legend()
        self.fig.suptitle("Displacement vectors overview")

    def disp_model_seq(self):
        """
        All functions to call to display the comparative displacement graphs
        """
        self.fig_clear()
        self.is_clear = False
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        self.ax1 = self.fig.add_subplot(gs[0])
        if self.area.dc.disp_data is None:
            self.plot_model(self.ax1)
        for sld in self.area.model.corresponding_slides:
            self.plot_slbl(self.ax1, self.area.slides[sld].slbl.slbl_surf, ratio=self.area.slides[sld].disp_ratio)
        self.plot_dem(self.ax1)
        if self.area.dc.disp_data is not None:
            self.plot_model_and_los(self.ax1)
            # self.add_los_on_model(self.ax1)
            self.ax2 = self.fig.add_subplot(gs[1], sharex=self.ax1)
            self.plot_moving_mean(self.ax2)
            self.ax2.set_xlabel("Distance (m)")
            self.ax2.set_ylabel("Data moving mean (50m)")
        else:
            self.ax1.set_xlabel("Distance (m)")
        self.ax1.set_ylabel("Elevation (m)")
        self.ax1.axis(self.model_axis)
        self.fig.subplots_adjust(hspace=0.3)
        self.fig.legend()
        self.fig.suptitle("Displacement model")

    def comp_disp_seq(self):
        """
        All functions to call to display the comparative displacement graphs
        """
        self.fig_clear()
        self.is_clear = False
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        self.plot_disp(self.ax1)
        self.plot_diff(self.ax2)
        self.ax1.set_ylabel("Displacement (cm)")
        self.ax2.set_xlabel("Distance (m)")
        self.ax2.set_ylabel("Displacement error (cm)")
        self.fig.subplots_adjust(hspace=0.3)
        self.fig.legend()
        self.fig.suptitle('RMSE = ' + f"{self.area.dc.rmse:.6f}" + ", r² = " + f"{self.area.dc.r2:.6f}")

    def corr_disp_seq(self):
        """
        All functions to call to display the correlation graph
        """
        self.fig_clear()
        self.is_clear = False
        self.ax = self.fig.add_subplot(111)
        self.plot_corr(self.ax)
        self.ax.set_xlabel("Data")
        self.ax.set_ylabel("Model")
        self.fig.legend()
        self.fig.suptitle("Correlation between displacement values viewed in the LOS in Data and Model")
