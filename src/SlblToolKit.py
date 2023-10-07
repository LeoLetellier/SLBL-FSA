"""
Python file containing useful and non-specific functions
"""

import pandas as pd
import numpy as np
from math import sin, cos, pi


def read_data_file(data_path, column1, column2=None):
    """
    Read the input file with Pandas
    :param data_path: input path file to data
    :param column1: first column to be red
    :param column2: range of columns to be red
    :return: pandas data frame
    """
    extension = data_path[data_path.find(".") + 1::]
    print("INFO:\tReading 2D data file. The file extension used is", extension)

    if extension == "xls" or extension == "xlsx":
        df = pd.read_excel(data_path)
    elif extension == "txt" or extension == "asc":
        # delimiter is " " or "\t"
        df = pd.read_csv(data_path, delim_whitespace=True)
    elif extension == "csv":
        # delimiter is ";"
        df = pd.read_csv(data_path, sep=';')
    else:
        df = None
        print(
            "!\tThe file extension '", extension,
            "' is not implemented in this script. Accepted extensions are :\n\t\txls  xlsx  txt  asc  csv")
    if column2 is None:
        return df.iloc[:, column1]
    return df.iloc[:, column1:column2 + 1]


def save_csv_file(data_path, data):
    """
    Save data to csv file
    :param data_path: path to where the file should be saved
    :param data: dict with column name as key and array as value
    :return: None
    """
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False, sep=";")


def intersection_on_topo(x, topo, xx, zz):
    """
    Search for an intersection point between a polyline and a segment
    :param x: x coordinates of the polyline
    :param topo: y coordinates of the polyline
    :param xx: x coordinates of the segment
    :param zz: y coordinates of the segment
    :return: coordinates of the intersection point if exists [x,y]
    """
    # progressing onto each subsegment of the polyline until finding the intersection if existing
    for k in range(np.shape(x)[0] - 1):
        if xx[0] <= x[k] <= xx[1]:
            intersect = get_intersection_point([x[k], x[k + 1]], [topo[k], topo[k + 1]], xx, zz)
            if intersect is not None:
                return intersect
    return None


def get_intersection_point(xk, zk, xx, zz):
    """
    Search for an intersection between two segments
    :param xk: x coordinates of the first segment
    :param zk: y coordinates of the first segment
    :param xx: x coordinates of the second segment
    :param zz: y coordinates of the second segment
    :return: coordinates of the intersection point if exists [x,y]
    """
    x1, y1, x2, y2 = xk[0], zk[0], xk[1], zk[1]
    x3, y3, x4, y4 = xx[0], zz[0], xx[1], zz[1]

    denominator = (x1 - x2) * (y3 - y4) - (x3 - x4) * (y1 - y2)
    if denominator == 0:  # segments are parallel
        return None
    intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    # Check that the solution belongs to the two segments (not out of range)
    if (min(x1, x2) <= intersection_x <= max(x1, x2) and min(y1, y2) <= intersection_y <= max(y1, y2) and
            min(x3, x4) <= intersection_x <= max(x3, x4) and min(y3, y4) <= intersection_y <= max(y3, y4)):
        return [intersection_x, intersection_y]
    return None


def check_regular_spacing(x_data, eps=0.01):
    """
    Check if the given data is regularly spaced
    :param x_data: data whose regularity has to be checked
    :param eps: tolerance
    :return: spacing used if regularly spaced, spacing_array otherwise
    """
    spacing_arr = np.empty((np.shape(x_data)[0] - 1))
    spacing_ref = abs(x_data[1] - x_data[0])  # first spacing taken as reference
    spacing_arr[0] = spacing_ref
    for k in range(2, np.shape(x_data)[0]):
        spacing = abs(x_data[k] - x_data[k - 1])
        spacing_arr[k - 1] = spacing
        if abs(spacing - spacing_ref) > eps:
            print("not regularly spaced")
            return spacing_arr  # empty after break index
    print("regularly spaced")
    return spacing_ref


def normal_vector_los(theta, delta):
    """
    Compute the normal vector associated to a LOS
    :param theta: angle between 3D vector and z-axis
    :param delta: angle between the xy projection and the North
    :return: normalized normal vector
    """
    t = np.deg2rad(theta)
    d = np.deg2rad(delta)
    vec = np.array([sin(t) * cos(d), sin(t) * sin(d), - cos(t)])
    return vec / np.linalg.norm(vec)


def normal_vector_cross_section(alpha):
    """
    Compute the normal vector of a cross-section
    :param alpha: angle between the cross-section direction and the North
    :return: normalized normal vector
    """
    alpha = alpha * pi / 180
    vec = np.array([cos(alpha), sin(alpha), 0])
    return vec / np.linalg.norm(vec)


def meter_2_idx(meter, data):
    """
    Return the first index following the searching value on a given axis
    :param meter: position on x-axis (float)
    :param data: x-axis data (np.ndarray)
    :return: index (integer)
    """
    for i in range(np.shape(data)[0]):
        if data[i] >= meter:
            break
    return i


def meter_2_idx_before(meter, data):
    """
    Return the index if equal, the one before if exceed
    :param meter: position on x-axis (float)
    :param data: x-axis data (np.ndarray)
    :return: index (integer)
    """
    for i in range(np.shape(data)[0]):
        if data[i] > meter:
            break
    return i - 1


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize an array of vectors
    :param vectors: array of vectors not normalized
    :return: array of normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def interpolation_moving_mean(x_sample: np.ndarray, value_sample: np.ndarray, x_target: np.ndarray,
                              window=50.) -> np.ndarray:
    value_target = np.zeros(shape=x_target.shape, dtype=float)
    for pnt in range(x_target.shape[0]):
        values = value_sample[np.abs(x_sample-x_target[pnt]) <= window]
        if values.shape[0] != 0:
            value_target[pnt] = np.mean(values)
        else:
            value_target[pnt] = np.nan
    return value_target
