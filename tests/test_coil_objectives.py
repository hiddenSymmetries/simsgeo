from simsgeo import JaxStelleratorSymmetricCylindricalFourierCurve, StelleratorSymmetricCylindricalFourierCurve, CurveLength, LpCurveCurvature, LpCurveTorsion, FourierCurve, MinimumDistance, JaxFourierCurve
import numpy as np
np.random.seed(1)
import pytest
from simsgeo import parameters
parameters['jit'] = True

def get_coil(curve, rand_scale=0.01):
    order = 10
    nquadpoints = 200
    coil = StelleratorSymmetricCylindricalFourierCurve(nquadpoints, order, 2)

    if curve == "FourierCurve":
        coil = FourierCurve(nquadpoints, order)
    elif curve == "JaxFourierCurve":
        coil = JaxFourierCurve(nquadpoints, order)
    elif curve == "StelleratorSymmetricCylindricalFourierCurve":
        coil = StelleratorSymmetricCylindricalFourierCurve(nquadpoints, order, 2)
    elif curve == "JaxStelleratorSymmetricCylindricalFourierCurve":
        coil = JaxStelleratorSymmetricCylindricalFourierCurve(nquadpoints, order, 2)
    else:
        assert False
    dofs = np.zeros((coil.num_dofs(), ))
    if curve in ["FourierCurve", "JaxFourierCurve"]:
        dofs[1] = 1.
        dofs[2*order+3] = 1.
        dofs[4*order+3] = 1.
    elif curve in ["StelleratorSymmetricCylindricalFourierCurve", "JaxStelleratorSymmetricCylindricalFourierCurve"]:
        dofs[0] = 1.
        dofs[1] = 0.1
        dofs[order+1] = 0.1
    else:
        assert False
    coil.set_dofs(dofs)

    dofs = np.asarray(coil.get_dofs())
    coil.set_dofs(dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape))
    return coil


@pytest.mark.parametrize("curve", ["FourierCurve", "JaxFourierCurve", "JaxStelleratorSymmetricCylindricalFourierCurve", "StelleratorSymmetricCylindricalFourierCurve"])
def test_curve_length_taylor_test(curve):
    coil = get_coil(curve)
    J = CurveLength(coil)
    J0 = J.J()
    coil_dofs = np.asarray(coil.get_dofs())
    h = 1e-3 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    dJ = J.dJ()
    deriv = np.sum(dJ * h)
    err = 1e6
    for i in range(5, 15):
        eps = 0.5**i
        coil.set_dofs(coil_dofs + eps * h)
        Jh = J.J()
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new


@pytest.mark.parametrize("curve", ["FourierCurve", "JaxFourierCurve", "JaxStelleratorSymmetricCylindricalFourierCurve", "StelleratorSymmetricCylindricalFourierCurve"])
def test_curve_curvature_taylor_test(curve):
    coil = get_coil(curve)
    J = LpCurveCurvature(coil, p=2)
    J0 = J.J()
    coil_dofs = np.asarray(coil.get_dofs())
    h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    dJ = J.dJ()
    deriv = np.sum(dJ * h)
    assert np.abs(deriv) > 1e-10
    err = 1e6
    for i in range(5, 15):
        eps = 0.5**i
        coil.set_dofs(coil_dofs + eps * h)
        Jh = J.J()
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new


@pytest.mark.parametrize("curve", ["FourierCurve", "JaxFourierCurve", "JaxStelleratorSymmetricCylindricalFourierCurve", "StelleratorSymmetricCylindricalFourierCurve"])
def test_curve_torsion_taylor_test(curve):
    coil = get_coil(curve)
    J = LpCurveTorsion(coil, p=2)
    J0 = J.J()
    coil_dofs = np.asarray(coil.get_dofs())
    h = 1e-3 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    dJ = J.dJ()
    deriv = np.sum(dJ * h)
    assert np.abs(deriv) > 1e-10
    err = 1e6
    for i in range(10, 20):
        eps = 0.5**i
        coil.set_dofs(coil_dofs + eps * h)
        Jh = J.J()
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new

@pytest.mark.parametrize("curve", ["FourierCurve", "JaxFourierCurve", "JaxStelleratorSymmetricCylindricalFourierCurve", "StelleratorSymmetricCylindricalFourierCurve"])
def test_curve_minimum_distance_taylor_test(curve):
    ncoils = 3
    coils = [get_coil(curve) for i in range(ncoils)]
    J = MinimumDistance(coils, 0.2)
    for k in range(ncoils):
        coil_dofs = np.asarray(coils[k].get_dofs())
        h = 1e-3 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
        J0 = J.J()
        dJ = J.dJ()[k]
        deriv = np.sum(dJ * h)
        assert np.abs(deriv) > 1e-10
        err = 1e6
        for i in range(5, 15):
            eps = 0.5**i
            coils[k].set_dofs(coil_dofs + eps * h)
            Jh = J.J()
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-deriv)
            print("err_new %s" % (err_new))
            assert err_new < 0.55 * err
            err = err_new

if __name__ == "__main__":
    test_curve_length_taylor_test()
