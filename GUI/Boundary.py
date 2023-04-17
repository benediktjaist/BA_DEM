import sympy as smp
import numpy as np

class Boundary:

    def __init__(self, start_point, end_point):
        self.start_point = np.array([start_point[0], start_point[1], 0])
        self.end_point = np.array([end_point[0], end_point[1], 0])
        self.id = None

    def calc_gradient(self):
        return (self.end_point[1] - self.start_point[1]) / (self.end_point[0] - self.start_point[0])

    def calc_axis_section(self):
        return self.start_point[1] - self.calc_gradient() * self.start_point[0]

    def get_lin_eq(self):
        x = smp.symbols('x')
        m = self.calc_gradient()
        t = self.calc_axis_section()
        return m * x + t

    def get_vert_line(self):
        return self.start_point[0]
