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
    mpol = 4
    ntor = 3
    nfp = 2
    stellsym = True
    phis = np.linspace(0, 1, 31, endpoint=False)
    thetas = np.linspace(0, 1, 31, endpoint=False)
    if surfacetype == "SurfaceRZFourier":
        from simsgeo import SurfaceRZFourier
        s = SurfaceRZFourier(mpol, ntor, nfp, stellsym, phis, thetas)
        s.rc[0, ntor + 0] = 1
        s.rc[1, ntor + 0] = 0.3
        s.zs[1, ntor + 0] = 0.3
    elif surfacetype == "SurfaceXYZFourier":
        from simsgeo import SurfaceXYZFourier
        stellsym = False
        s = SurfaceXYZFourier(mpol, ntor, nfp, stellsym, phis, thetas)
        s.xc[0, ntor + 1] = 1.
        s.xc[1, ntor + 1] = 0.1
        s.ys[0, ntor + 1] = 1.
        s.ys[1, ntor + 1] = 0.1
        s.zs[1, ntor] = 0.1
    else:
        assert False

    dofs = np.asarray(s.get_dofs())
    np.random.seed(2)
    rand_scale=0.01
    s.set_dofs(dofs + rand_scale * np.random.rand(len(dofs)).reshape(dofs.shape))
    return s


class Testing(unittest.TestCase):

    surfacetypes = ["SurfaceRZFourier", "SurfaceXYZFourier"]

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


    def subtest_surface_normal_coefficient_derivative(self, s):
        coeffs = s.get_dofs()
        s.invalidate_cache()
        def f(dofs):
            s.set_dofs(dofs)
            return s.normal()[1, 1, :].copy()
        def df(dofs):
            s.set_dofs(dofs)
            return s.dnormal_by_dcoeff()[1, 1, :, :].copy()
        taylor_test(f, df, coeffs)


    def test_surface_normal_coefficient_derivative(self):
        for surfacetype in self.surfacetypes:
                with self.subTest(surfacetype=surfacetype):
                    s = get_surface(surfacetype)
                    self.subtest_surface_normal_coefficient_derivative(s)

    def subtest_surface_area_coefficient_derivative(self, s):
        coeffs = s.get_dofs()
        s.invalidate_cache()
        def f(dofs):
            s.set_dofs(dofs)
            return np.asarray(s.surface_area())
        def df(dofs):
            s.set_dofs(dofs)
            return s.dsurface_area_by_dcoeff()[None, :].copy()
        taylor_test(f, df, coeffs)


    def test_surface_area_coefficient_derivative(self):
        for surfacetype in self.surfacetypes:
                with self.subTest(surfacetype=surfacetype):
                    s = get_surface(surfacetype)
                    self.subtest_surface_area_coefficient_derivative(s)

if __name__ == "__main__":
    print('wtf')
    unittest.main()
