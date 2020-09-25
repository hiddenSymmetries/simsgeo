from simsgeo import StelleratorSymmetricCylindricalFourierCurve, CurveLength, LpCurveCurvature, LpCurveTorsion, FourierCurve, MinimumDistance
import numpy as np
np.random.seed(1)
from simsgeo import parameters
parameters['jit'] = True

def get_coil(rand_scale=0.01):
    order = 10
    nquadpoints = 200
    coil = StelleratorSymmetricCylindricalFourierCurve(nquadpoints, order, 2)
    coil.coefficients[0][0] = 1.
    coil.coefficients[0][1] = 0.1
    coil.coefficients[1][0] = 0.1

    # coil = FourierCurve(nquadpoints, order)
    # dofs = np.zeros((coil.num_dofs(), ))
    # dofs[1] = 1.
    # dofs[2*order+3] = 1.
    # dofs[4*order+3] = 1.
    # coil.set_dofs(dofs)

    dofs = np.asarray(coil.get_dofs())
    coil.set_dofs(dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape))
    return coil


def test_curve_length_taylor_test():
    coil = get_coil()
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
        coil.invalidate_cache()
        Jh = J.J()
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new


def test_curve_curvature_taylor_test():
    coil = get_coil()
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
        coil.invalidate_cache()
        Jh = J.J()
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new


def test_curve_torsion_taylor_test():
    coil = get_coil()
    J = LpCurveTorsion(coil, p=2)
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
        coil.invalidate_cache()
        Jh = J.J()
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-deriv)
        print("err_new %s" % (err_new))
        assert err_new < 0.55 * err
        err = err_new

def test_curve_minimum_distance_taylor_test():
    ncoils = 3
    coils = [get_coil() for i in range(ncoils)]
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
            coils[k].invalidate_cache()
            Jh = J.J()
            deriv_est = (Jh-J0)/eps
            err_new = np.linalg.norm(deriv_est-deriv)
            print("err_new %s" % (err_new))
            assert err_new < 0.55 * err
            err = err_new

if __name__ == "__main__":
    test_curve_minimum_distance_taylor_test()
