import math
import random
import copy
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

from cellbin.contrib import Line

model = LinearRegression()


def random_color():
    b = random.randint(0, 256)
    g = random.randint(0, 256)
    r = random.randint(0, 256)
    return b, g, r


class TrackLineDetector(object):
    def __init__(self):
        self.grid = 100

    def generate(self, arr):
        """
        This algorithm will not work the angle of image is more than 8 degree

        Args:
            arr (): 2D array in uint 8 or uint 16

        Returns:
            h_lines: horizontal line
            v_lines: vertical line

        """
        h_lines, v_lines = [], []

        # horizontal direction
        horizontal_candidate_pts = self.create_candidate_pts(arr, 'x')
        h_angle = self.integer_angle(horizontal_candidate_pts, 'x')
        if h_angle != -1000:
            horizontal_pts = self.select_pts_by_integer_angle(horizontal_candidate_pts, h_angle, tolerance=1)
            if len(horizontal_pts) != 0:
                horizontal_color_pts = self.classify_points(horizontal_pts, h_angle, tolerance=1)
                h_lines = self.points_to_line(horizontal_color_pts, tolerance=3)

        # vertical direction
        vertical_candidate_pts = self.create_candidate_pts(arr, 'y')
        v_angle = self.integer_angle(vertical_candidate_pts, 'y')
        if v_angle != -1000:
            vertical_pts = self.select_pts_by_integer_angle(vertical_candidate_pts, v_angle, tolerance=1)
            if len(vertical_pts) != 0:
                vertical_color_pts = self.classify_points(vertical_pts, v_angle, tolerance=1)
                v_lines = self.points_to_line(vertical_color_pts, tolerance=3)

        return h_lines, v_lines

    @staticmethod
    def points_to_line(dct, tolerance=2):
        lines = list()
        for k, v in dct.items():
            if len(v) > tolerance:
                tmp = np.array(v)
                model.fit(tmp[:, 0].reshape(-1, 1), tmp[:, 1]) 
                line = Line()
                line.init_by_point_k(v[0], model.coef_[0])
                lines.append(line)
        return lines

    def classify_points(self, candidate_pts, base_angle, tolerance=2):
        pts = copy.copy(candidate_pts)
        ind = 0
        dct = dict()
        while (len(pts) > 1):
            pts_, index = self.angle_line(base_angle, pts, tolerance)
            dct[ind] = pts_
            pts = np.delete(np.array(pts), index, axis=0).tolist()
            ind += 1
        return dct

    @staticmethod
    def angle_line(angle, points, tolerance=2):
        count = len(points)
        orignal_point = points[0]
        points_ = [points[0]]
        index = [0]
        for i in range(1, count):
            p = points[i]
            line = Line()
            line.init_by_point_pair(orignal_point, p)
            diff = abs(line.rotation() - angle)
            diff = (diff > 90) and (180 - diff) or diff
            if diff < tolerance:
                points_.append(p)
                index.append(i)
        return points_, index

    @staticmethod
    def select_pts_by_integer_angle(candidate_pts, base_angle, tolerance=2):
        x_count = len(candidate_pts)
        pts = list()
        for i in range(0, x_count - 1):
            if len(candidate_pts[i]) > 100:
                continue
            pts_start = candidate_pts[i]
            pts_end = candidate_pts[i + 1]
            for p0 in pts_start:
                d = [math.sqrt(pow(p0[0] - p1[0], 2) + pow(p0[1] - p1[1], 2)) for p1 in pts_end]
                d_ = np.abs(d)
                ind = np.where(d_ == np.min(d_))[0]
                line = Line()
                line.init_by_point_pair(p0, pts_end[ind[0]])
                if abs(line.rotation() - base_angle) <= tolerance: pts.append(p0)
        return pts

    @staticmethod
    def integer_angle(pts, derection='x'):
        angle = -1000
        x_count = len(pts)
        angles = list()
        for i in range(0, x_count - 1):
            if len(pts[i]) > 100:
                continue
            pts_start = pts[i]
            pts_end = pts[i + 1]
            for p0 in pts_start:
                d = [math.sqrt(pow(p0[0] - p1[0], 2) + pow(p0[1] - p1[1], 2)) for p1 in pts_end]
                d_ = np.abs(d)
                ind = np.where(d_ == np.min(d_))[0]
                line = Line()
                line.init_by_point_pair(p0, pts_end[ind[0]])
                angles.append(round(line.rotation()))
        if len(angles) != 0:
            x = np.array(angles) - np.min(angles)
            angle = np.argmax(np.bincount(x)) + np.min(angles)
        return angle

    def create_candidate_pts(self, mat, derection='x'):
        pts = list()
        h, w = mat.shape
        # direction x -> h
        # direction y -> w
        counter = (derection == 'x' and h or w)
        for i in range(0, counter, self.grid):
            t = i + self.grid / 2
            if derection == 'x':
                region_mat = mat[i: i + self.grid, :w]
                if region_mat.shape[0] != self.grid:
                    continue
                line = np.sum(region_mat, axis=0) / self.grid
            else:
                region_mat = mat[:h, i: i + self.grid]
                if region_mat.shape[1] != self.grid:
                    continue
                line = np.sum(region_mat, axis=1) / self.grid
            p = argrelextrema(line, np.less_equal, order=100)
            # print(p[0].shape)
            if derection == 'x':
                pt = [[p, t] for p in p[0]]
            else:
                pt = [[t, p] for p in p[0]]
            pts.append(pt)
        return pts


def main():
    import cv2
    image_path = r"D:\Data\tmp\Y00035MD\Y00035MD\Y00035MD_0000_0004_2023-01-30_15-50-41-868.tif"
    arr = cv2.imread(image_path, -1)
    ftl = TrackLineDetector()
    result = ftl.generate(arr)


if __name__ == '__main__':
    main()
