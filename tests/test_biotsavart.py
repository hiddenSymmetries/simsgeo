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
