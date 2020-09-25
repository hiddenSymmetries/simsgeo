import numpy as np
import simsgeopp as sgpp
from jax import grad, vjp, jacfwd, jvp
from .jit import jit
import jax.numpy as jnp


@jit
def kappa_pure(d1gamma, d2gamma):
    return jnp.linalg.norm(jnp.cross(d1gamma, d2gamma), axis=1)/jnp.linalg.norm(d1gamma, axis=1)**3

kappagrad0 = jit(lambda d1gamma, d2gamma, v: vjp(lambda d1g: kappa_pure(d1g, d2gamma), d1gamma)[1](v))
kappagrad1 = jit(lambda d1gamma, d2gamma, v: vjp(lambda d2g: kappa_pure(d1gamma, d2g), d2gamma)[1](v))


@jit
def torsion_pure(d1gamma, d2gamma, d3gamma):
    return jnp.sum(jnp.cross(d1gamma, d2gamma, axis=1) * d3gamma, axis=1) / jnp.sum(jnp.cross(d1gamma, d2gamma, axis=1)**2, axis=1)

torsiongrad0 = jit(lambda d1gamma, d2gamma, d3gamma, v: vjp(lambda d1g: torsion_pure(d1g, d2gamma, d3gamma), d1gamma)[1](v))
torsiongrad1 = jit(lambda d1gamma, d2gamma, d3gamma, v: vjp(lambda d2g: torsion_pure(d1gamma, d2g, d3gamma), d2gamma)[1](v))
torsiongrad2 = jit(lambda d1gamma, d2gamma, d3gamma, v: vjp(lambda d3g: torsion_pure(d1gamma, d2gamma, d3g), d3gamma)[1](v))

class Curve(object):

    def plot(self, ax=None, show=True, plot_derivative=False, closed_loop=True, color=None, linestyle=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        gamma = self.gamma()
        gammadash = self.gammadash()
        if ax is None:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

        def rep(data):
            if closed_loop:
                return np.concatenate((data, [data[0]]))
            else:
                return data
        ax.plot(rep(gamma[:, 0]), rep(gamma[:, 1]), rep(
            gamma[:, 2]), color=color, linestyle=linestyle)
        if plot_derivative:
            ax.quiver(rep(gamma[:, 0]), rep(gamma[:, 1]), rep(gamma[:, 2]), 0.1 * rep(gammadash[:, 0]),
                      0.1 * rep(gammadash[:, 1]), 0.1 * rep(gammadash[:, 2]), arrow_length_ratio=0.1, color="r")
        if show:
            plt.show()
        return ax

    def kappa(self):
        return np.asarray(kappa_pure(self.gammadash(), self.gammadashdash()))

    def torsion(self):
        return np.asarray(torsion_pure(self.gammadash(), self.gammadashdash(), self.gammadashdashdash()))

    def dkappa_by_dcoeff_vjp(self, v):
        return self.dgammadash_by_dcoeff_vjp(kappagrad0(self.gammadash(), self.gammadashdash(), v)) \
            + self.dgammadashdash_by_dcoeff_vjp(kappagrad1(self.gammadash(), self.gammadashdash(), v))

    def dtorsion_by_dcoeff_vjp(self, v):
        return self.dgammadash_by_dcoeff_vjp(torsiongrad0(self.gammadash(), self.gammadashdash(), self.gammadashdashdash(), v)) \
            + self.dgammadashdash_by_dcoeff_vjp(torsiongrad1(self.gammadash(), self.gammadashdash(), self.gammadashdashdash(), v)) \
            + self.dgammadashdashdash_by_dcoeff_vjp(torsiongrad2(self.gammadash(), self.gammadashdash(), self.gammadashdashdash(), v))

class JaxCurve(sgpp.Curve, Curve):
    def __init__(self, numquadpoints, gamma_pure):
        sgpp.Curve.__init__(self, numquadpoints)
        self.gamma_pure = gamma_pure
        points = np.asarray(self.quadpoints)
        ones = jnp.ones_like(points)

        self.gamma_jax = jit(lambda dofs: self.gamma_pure(dofs, points))
        self.dgamma_by_dcoeff_jax = jit(jacfwd(self.gamma_jax))
        self.dgamma_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.gamma_jax, x)[1](v))

        self.gammadash_pure = lambda x, q: jvp(lambda p: self.gamma_pure(x, p), (q,), (ones,))[1]
        self.gammadash_jax = jit(lambda x: self.gammadash_pure(x, points))
        self.dgammadash_by_dcoeff = jit(jacfwd(self.gammadash_jax))
        self.dgammadash_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.gammadash_jax, x)[1](v))

        self.gammadashdash_pure = lambda x, q: jvp(lambda p: self.gammadash_pure(x, p), (q,), (ones,))[1]
        self.gammadashdash_jax = jit(lambda x: self.gammadashdash_pure(x, points))
        self.dgammadashdash_by_dcoeff_jax = jit(jacfwd(self.gammadashdash_jax))
        self.dgammadashdash_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.gammadashdash_jax, x)[1](v))

        self.gammadashdashdash_pure = lambda x, q: jvp(lambda p: self.gammadashdash_pure(x, p), (q,), (ones,))[1]
        self.gammadashdashdash_jax = jit(lambda x: self.gammadashdashdash_pure(x, points))
        self.dgammadashdashdash_by_dcoeff_jax = jit(jacfwd(self.gammadashdashdash_jax))
        self.dgammadashdashdash_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.gammadashdashdash_jax, x)[1](v))

        self.dkappa_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(lambda d: kappa_pure(self.gammadash_jax(d), self.gammadashdash_jax(d)), x)[1](v))
        self.dkappa_by_dcoeff_jax = jit(jacfwd(lambda d: kappa_pure(self.gammadash_jax(d), self.gammadashdash_jax(d))))

        self.dtorsion_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(lambda d: torsion_pure(self.gammadash_jax(d), self.gammadashdash_jax(d), self.gammadashdashdash_jax(d)), x)[1](v))

    def gamma_impl(self, gamma):
        gamma[:, :] = self.gamma_jax(self.get_dofs())

    def dgamma_by_dcoeff_impl(self, dgamma_by_dcoeff):
        dgamma_by_dcoeff[:, :, :] = self.dgamma_by_dcoeff_jax(self.get_dofs())

    def dgamma_by_dcoeff_vjp(self, v):
        return self.dgamma_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def gammadash_impl(self, gammadash):
        gammadash[:, :] = self.gammadash_jax(self.get_dofs())

    def dgammadash_by_dcoeff_vjp(self, v):
        return self.dgammadash_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:, :] = self.gammadashdash_jax(self.get_dofs())

    def dgammadashdash_by_dcoeff_vjp(self, v):
        return self.dgammadashdash_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:, :] = self.gammadashdashdash_jax(self.get_dofs())

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        return self.dgammadashdashdash_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def dkappa_by_dcoeff_vjp(self, v):
        return self.dkappa_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def dtorsion_by_dcoeff_vjp(self, v):
        return self.dtorsion_by_dcoeff_vjp_jax(self.get_dofs(), v)
