import numpy as np
import unittest
from simsgeo import parameters
parameters['jit'] = False

def taylor_test(f, df, x, epsilons=None, direction=None):
    np.random.seed(1)
    f0 = f(x)
    if direction is None:
        direction = np.random.rand(*(x.shape))-0.5
    dfx = df(x)@direction
    if epsilons is None:
        epsilons = np.power(2., -np.asarray(range(7, 20)))
    # print("################################################################################")
    err_old = 1e9
    counter = 0
    for eps in epsilons:
        if counter > 8:
            break
        fpluseps = f(x + eps * direction)
        fminuseps = f(x - eps * direction)
        dfest = (fpluseps-fminuseps)/(2*eps)
        err = np.linalg.norm(dfest - dfx)
        print(err)
        assert err < 1e-9 or err < 0.3 * err_old
        if err < 1e-9:
            break
        err_old = err
        counter += 1
    if err > 1e-10:
        assert counter > 3
    # print("################################################################################")

def get_surface(surfacetype):
    mpol = 2
    ntor = 1
    nfp = 1
    stellsym = True
    if surfacetype == "SurfaceRZFourier":
        from simsgeo import SurfaceRZFourier
        s = SurfaceRZFourier(mpol, ntor, nfp, stellsym, np.linspace(0, 1, 31, endpoint=False), np.linspace(0, 1, 31, endpoint=False))
        
    else:
        assert False

    if surfacetype in ["SurfaceRZFourier"]:
        s.rc[0, ntor + 0] = 1
        s.rc[1, ntor + 0] = 0.3
        s.zs[1, ntor + 0] = 0.3
    else:
        assert False

    dofs = np.asarray(s.get_dofs())
    np.random.seed(2)
    rand_scale=0.01
    s.set_dofs(dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape))
    return s


class Testing(unittest.TestCase):

    surfacetypes = ["SurfaceRZFourier"]

    def subtest_surface_coefficient_derivative(self, s):
        coeffs = s.get_dofs()
        s.invalidate_cache()
        def f(dofs):
            s.set_dofs(dofs)
            return s.gamma()[1, 1, :].copy()
        def df(dofs):
            s.set_dofs(dofs)
            return s.dgamma_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            s.set_dofs(dofs)
            return s.gammadash1()[1, 1, :].copy()
        def df(dofs):
            s.set_dofs(dofs)
            return s.dgammadash1_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs)

        def f(dofs):
            s.set_dofs(dofs)
            return s.gammadash2()[1, 1, :].copy()
        def df(dofs):
            s.set_dofs(dofs)
            return s.dgammadash2_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs)

    def test_surface_coefficient_derivative(self):
        for surfacetype in self.surfacetypes:
                with self.subTest(surfacetype=surfacetype):
                    s = get_surface(surfacetype)
                    self.subtest_surface_coefficient_derivative(s)


if __name__ == "__main__":
    print('wtf')
    unittest.main()
