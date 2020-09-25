from jax import grad, vjp, jit
import jax.numpy as jnp
import numpy as np

@jit
def curve_length_pure(gammadash):
    return jnp.mean(jnp.linalg.norm(gammadash, axis=1))

class CurveLength():

    def __init__(self, curve):
        self.curve = curve
        self.thisgrad = jit(lambda gammadash: grad(curve_length_pure)(gammadash))

    def J(self):
        return curve_length_pure(self.curve.gammadash())

    def dJ(self):
        return self.curve.dgammadash_by_dcoeff_vjp(self.thisgrad(self.curve.gammadash()))

@jit
def Lp_curvature_pure(kappa, gammadash, p, desired_kappa):
        arc_length = jnp.linalg.norm(gammadash, axis=1)
        return (1./p)*jnp.mean(jnp.maximum(kappa-desired_kappa, 0)**p * arc_length)

class LpCurveCurvature():

    def __init__(self, curve, p, desired_length=None):
        self.curve = curve
        if desired_length is None:
            self.desired_kappa = 0
        else:
            radius = desired_length/(2*pi)
            self.desired_kappa = 1/radius
        
        self.J_jax = jit(lambda kappa, gammadash: Lp_curvature_pure(kappa, gammadash, p, self.desired_kappa))
        self.thisgrad0 = jit(lambda kappa, gammadash: grad(self.J_jax, argnums=0)(kappa, gammadash))
        self.thisgrad1 = jit(lambda kappa, gammadash: grad(self.J_jax, argnums=1)(kappa, gammadash))

    def J(self):
        return self.J_jax(self.curve.kappa(), self.curve.gammadash())

    def dJ(self):
        grad0 = self.thisgrad0(self.curve.kappa(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.kappa(), self.curve.gammadash())
        return self.curve.dkappa_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

@jit
def Lp_torsion_pure(torsion, gammadash, p):
        arc_length = jnp.linalg.norm(gammadash, axis=1)
        return (1./p)*jnp.mean(jnp.abs(torsion)**p * arc_length)

class LpCurveTorsion():

    def __init__(self, curve, p):
        self.curve = curve
        
        self.J_jax = jit(lambda torsion, gammadash: Lp_torsion_pure(torsion, gammadash, p))
        self.thisgrad0 = jit(lambda torsion, gammadash: grad(self.J_jax, argnums=0)(torsion, gammadash))
        self.thisgrad1 = jit(lambda torsion, gammadash: grad(self.J_jax, argnums=1)(torsion, gammadash))

    def J(self):
        return self.J_jax(self.curve.torsion(), self.curve.gammadash())

    def dJ(self):
        grad0 = self.thisgrad0(self.curve.torsion(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.torsion(), self.curve.gammadash())
        return self.curve.dtorsion_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)
