from .curve import Curve
import numpy as np
import simsgeopp as sgpp
from math import pi

class StelleratorSymmetricCylindricalFourierCurve(sgpp.Curve, Curve):

    """ This class can for example be used to describe a magnetic axis. """

    def __init__(self, numquadpoints, nfp, order):
        sgpp.Curve.__init__(self, numquadpoints)
        self.coefficients = [np.zeros((order+1,)), np.zeros((order,))]
        self.nfp = nfp
        self.order = order


    def num_coeff(self):
        return 2*self.order+1

    def get_dofs(self):
        return np.concatenate(self.coefficients)

    def set_dofs(self, dofs):
        counter = 0
        for i in range(self.order+1):
            self.coefficients[0][i] = dofs[i]
        for i in range(self.order):
            self.coefficients[1][i] = dofs[self.order + 1 + i]

    def gamma_impl(self, gamma):
        points = np.asarray(self.quadpoints)
        nfp = self.nfp
        for i in range(self.order+1):
            gamma[:, 0] += self.coefficients[0][i] * np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            gamma[:, 1] += self.coefficients[0][i] * np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
        for i in range(1, self.order+1):
            gamma[:, 2] += self.coefficients[1][i-1] * np.sin(nfp * 2 * pi * i * points)

    def dgamma_by_dcoeff_impl(self, dgamma_by_dcoeff):
        points = np.asarray(self.quadpoints)
        nfp = self.nfp
        for i in range(self.order+1):
            dgamma_by_dcoeff[i, :, 0] = np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            dgamma_by_dcoeff[i, :, 1] = np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
        for i in range(1, self.order+1):
            dgamma_by_dcoeff[self.order + i, :, 2] = np.sin(nfp * 2 * pi * i * points)

    def dgamma_by_dphi_impl(self, dgamma_by_dphi):
        points = np.asarray(self.quadpoints)
        nfp = self.nfp
        for i in range(self.order+1):
            dgamma_by_dphi[0, :, 0] += self.coefficients[0][i] * (
                -(nfp * 2 * pi * i) * np.sin(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
                -(2 * pi) *           np.cos(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
            )
            dgamma_by_dphi[0, :, 1] += self.coefficients[0][i] * (
                -(nfp * 2 * pi * i) * np.sin(nfp * 2 * pi * i * points) * np.sin(2 * pi * points)
                +(2 * pi) *           np.cos(nfp * 2 * pi * i * points) * np.cos(2 * pi * points)
            )
        for i in range(1, self.order+1):
            dgamma_by_dphi[0, :, 2] += self.coefficients[1][i-1] * (nfp * 2 * pi * i) * np.cos(nfp * 2 * pi * i * points)
        return dgamma_by_dphi
