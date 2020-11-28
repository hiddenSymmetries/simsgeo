import numpy as np
import pytest
from simsgeo import FourierCurve, StelleratorSymmetricCylindricalFourierCurve, BiotSavart

def get_coil(num_quadrature_points=200):
    coil = FourierCurve(num_quadrature_points, 3)
    coeffs = coil.dofs
    coeffs[1][0] = 1.
    coeffs[1][1] = 0.5
    coeffs[2][2] = 0.5
    coil.set_dofs(np.concatenate(coeffs))
    return coil

def test_biotsavart_exponential_convergence():
    coil = get_coil()
    from time import time
    # points = np.asarray(17 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
    points = np.asarray(10 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
    tic = time()
    btrue   = BiotSavart([get_coil(1000)], [1e4]).set_points(points).B(compute_derivatives=0)
    print(btrue)
    bcoarse = BiotSavart([get_coil(10)]  , [1e4]).set_points(points).B(compute_derivatives=0)
    bfine   = BiotSavart([get_coil(20)]  , [1e4]).set_points(points).B(compute_derivatives=0)
    assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)
    print(time()-tic)

    tic = time()
    dbtrue   = BiotSavart([get_coil(1000)], [1e4]).set_points(points).dB_by_dX(compute_derivatives=1)
    print(dbtrue)
    dbcoarse = BiotSavart([get_coil(10)]  , [1e4]).set_points(points).dB_by_dX(compute_derivatives=1)
    dbfine   = BiotSavart([get_coil(20)]  , [1e4]).set_points(points).dB_by_dX(compute_derivatives=1)
    assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)
    print(time()-tic)

    tic = time()
    dbtrue   = BiotSavart([get_coil(1000)], [1e4]).set_points(points).d2B_by_dXdX(compute_derivatives=2)
    print("dbtrue", dbtrue)
    dbcoarse = BiotSavart([get_coil(10)]  , [1e4]).set_points(points).d2B_by_dXdX(compute_derivatives=2)
    dbfine   = BiotSavart([get_coil(20)]  , [1e4]).set_points(points).d2B_by_dXdX(compute_derivatives=2)
    assert np.linalg.norm(btrue-bfine) < 1e-4 * np.linalg.norm(bcoarse-bfine)
    print(time()-tic)

def test_dB_by_dcoilcoeff_reverse_taylortest():
    np.random.seed(1)
    coil = get_coil()
    bs = BiotSavart([coil], [1e4])
    points = np.asarray(17 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
    points += 0.001 * (np.random.rand(*points.shape)-0.5)

    bs.set_points(points)
    coil_dofs = np.asarray(coil.get_dofs())
    B = bs.B()
    dBdX = bs.dB_by_dX()
    J0 = np.sum(B**2)
    dJ = bs.B_and_dB_vjp(B, dBdX)

    h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    dJ_dh = 2*np.sum(dJ[0][0] * h)
    err = 1e6
    for i in range(5, 10):
        eps = 0.5**i
        coil.set_dofs(coil_dofs + eps * h)
        bs.clear_cached_properties()
        Bh = bs.B()
        Jh = np.sum(Bh**2)
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-dJ_dh)
        assert err_new < 0.55 * err
        err = err_new

def test_dBdX_by_dcoilcoeff_reverse_taylortest():
    np.random.seed(1)
    coil = get_coil()
    bs = BiotSavart([coil], [1e4])
    points = np.asarray(17 * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04 ]])
    points += 0.001 * (np.random.rand(*points.shape)-0.5)

    bs.set_points(points)
    coil_dofs = np.asarray(coil.get_dofs())
    B = bs.B()
    dBdX = bs.dB_by_dX()
    J0 = np.sum(dBdX**2)
    dJ = bs.B_and_dB_vjp(B, dBdX)

    h = 1e-2 * np.random.rand(len(coil_dofs)).reshape(coil_dofs.shape)
    dJ_dh = 2*np.sum(dJ[1][0] * h)
    err = 1e6
    for i in range(5, 10):
        eps = 0.5**i
        coil.set_dofs(coil_dofs + eps * h)
        bs.clear_cached_properties()
        dBdXh = bs.dB_by_dX()
        Jh = np.sum(dBdXh**2)
        deriv_est = (Jh-J0)/eps
        err_new = np.linalg.norm(deriv_est-dJ_dh)
        assert err_new < 0.55 * err
        err = err_new
