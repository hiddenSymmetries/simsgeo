from .curve import JaxCurve
from math import pi
from jax.ops import index, index_add
import jax.numpy as jnp
import numpy as np


def stelleratorsymmetriccylindricalfouriercurve_pure(dofs, quadpoints, order, nfp):
    coefficients0 = dofs[:(order+1)]
    coefficients1 = dofs[(order+1):]
    gamma = jnp.zeros((len(quadpoints), 3))
    for i in range(order+1):
        gamma = index_add(gamma, index[:, 0], coefficients0[i] * jnp.cos(nfp * 2 * pi * i * quadpoints) * jnp.cos(2 * pi * quadpoints))
        gamma = index_add(gamma, index[:, 1], coefficients0[i] * jnp.cos(nfp * 2 * pi * i * quadpoints) * jnp.sin(2 * pi * quadpoints))
    for i in range(1, order+1):
        gamma = index_add(gamma, index[:, 2], coefficients1[i-1] * jnp.sin(nfp * 2 * pi * i * quadpoints))
    return gamma


class StelleratorSymmetricCylindricalFourierCurve(JaxCurve):

    """ This class can for example be used to describe a magnetic axis. """

    def __init__(self, numquadpoints, order, nfp):
        pure = lambda dofs, points: stelleratorsymmetriccylindricalfouriercurve_pure(dofs, points, order, nfp)
        super().__init__(numquadpoints, pure)
        self.order = order
        self.nfp = nfp
        self.coefficients = [np.zeros((order+1,)), np.zeros((order,))]

    def num_dofs(self):
        return 2*self.order+1

    def get_dofs(self):
        return np.concatenate(self.coefficients)

    def set_dofs(self, dofs):
        counter = 0
        for i in range(self.order+1):
            self.coefficients[0][i] = dofs[i]
        for i in range(self.order):
            self.coefficients[1][i] = dofs[self.order + 1 + i]
