from simsgeo import CurveXYZFourier, RotatedCurve, BiotSavart, boozer_surface_residual, SurfaceRZFourier, CurveRZFourier
import numpy as np
from math import pi

class CoilCollection():
    """
    Given some input coils and currents, this performs the reflection and
    rotation to generate a full set of stellarator coils.
    """

    def __init__(self, coils, currents, nfp, stellarator_symmetry):
        self._base_coils = coils
        self._base_currents = currents
        self.coils = []
        self.currents = []
        flip_list = [False, True] if stellarator_symmetry else [False] 
        self.map = []
        self.current_sign = []
        for k in range(0, nfp):
            for flip in flip_list:
                for i in range(len(coils)):
                    if k == 0 and not flip:
                        self.coils.append(self._base_coils[i])
                        self.currents.append(self._base_currents[i])
                    else:
                        rotcoil = RotatedCurve(coils[i], 2*pi*k/nfp, flip)
                        self.coils.append(rotcoil)
                        self.currents.append(-self._base_currents[i] if flip else currents[i])
                    self.map.append(i)
                    self.current_sign.append(-1 if flip else +1)
        dof_ranges = [(0, len(self._base_coils[0].get_dofs()))]
        for i in range(1, len(self._base_coils)):
            dof_ranges.append((dof_ranges[-1][1], dof_ranges[-1][1] + len(self._base_coils[i].get_dofs())))
        self.dof_ranges = dof_ranges

def get_ncsx_data(Nt_coils=25, Nt_ma=10, ppp=10):
    coil_data = np.loadtxt("NCSX_coil_coeffs.dat", delimiter=',')
    nfp = 3
    num_coils = 3
    coils = [CurveXYZFourier(Nt_coils*ppp, Nt_coils) for i in range(num_coils)]
    for ic in range(num_coils):
        dofs = coils[ic].dofs
        dofs[0][0] = coil_data[0, 6*ic + 1]
        dofs[1][0] = coil_data[0, 6*ic + 3]
        dofs[2][0] = coil_data[0, 6*ic + 5]
        for io in range(0, Nt_coils):
            dofs[0][2*io+1] = coil_data[io+1, 6*ic + 0]
            dofs[0][2*io+2] = coil_data[io+1, 6*ic + 1]
            dofs[1][2*io+1] = coil_data[io+1, 6*ic + 2]
            dofs[1][2*io+2] = coil_data[io+1, 6*ic + 3]
            dofs[2][2*io+1] = coil_data[io+1, 6*ic + 4]
            dofs[2][2*io+2] = coil_data[io+1, 6*ic + 5]
        coils[ic].set_dofs(np.concatenate(dofs))

    currents = [6.52271941985300E+05, 6.51868569367400E+05, 5.37743588647300E+05]
    currents = [c/1.474 for c in currents] # normalise to get a magnetic field of around 1 at the axis
    cR = [1.471415400740515, 0.1205306261840785, 0.008016125223436036, -0.000508473952304439, -0.0003025251710853062, -0.0001587936004797397, 3.223984137937924e-06, 3.524618949869718e-05, 2.539719080181871e-06, -9.172247073731266e-06, -5.9091166854661e-06, -2.161311017656597e-06, -5.160802127332585e-07, -4.640848016990162e-08, 2.649427979914062e-08, 1.501510332041489e-08, 3.537451979994735e-09, 3.086168230692632e-10, 2.188407398004411e-11, 5.175282424829675e-11, 1.280947310028369e-11, -1.726293760717645e-11, -1.696747733634374e-11, -7.139212832019126e-12, -1.057727690156884e-12, 5.253991686160475e-13]
    sZ = [0.06191774986623827, 0.003997436991295509, -0.0001973128955021696, -0.0001892615088404824, -2.754694372995494e-05, -1.106933185883972e-05, 9.313743937823742e-06, 9.402864564707521e-06, 2.353424962024579e-06, -1.910411249403388e-07, -3.699572817752344e-07, -1.691375323357308e-07, -5.082041581362814e-08, -8.14564855367364e-09, 1.410153957667715e-09, 1.23357552926813e-09, 2.484591855376312e-10, -3.803223187770488e-11, -2.909708414424068e-11, -2.009192074867161e-12, 1.775324360447656e-12, -7.152058893039603e-13, -1.311461207101523e-12, -6.141224681566193e-13, -6.897549209312209e-14]

    numpoints = Nt_ma*ppp+1 if ((Nt_ma*ppp) % 2 == 0) else Nt_ma*ppp
    ma = CurveRZFourier(numpoints, Nt_ma, nfp, True)
    ma.rc[:] = cR[0:(Nt_ma+1)]
    ma.zs[:] = sZ[0:Nt_ma]
    return (coils, currents, ma)


coils, currents, ma = get_ncsx_data()
stellarator = CoilCollection(coils, currents, 3, True)
bs = BiotSavart(stellarator.coils, stellarator.currents)
iota = -0.3

mpol = 5
ntor = 5

nfp = 1
stellsym = False
nphi = 30
ntheta = 30

phis = np.linspace(0, 1, nphi, endpoint=False)
thetas = np.linspace(0, 1, ntheta, endpoint=False)
from simsgeo import SurfaceXYZFourier
# this is a donut with major radius 1 and minor radius 0.1 probably a terrible
# initial guess and we should put a tube around the magnetic axis instead this
# could be done by solving a least squares problem 
# A^T A dofs = A^T b
# where b are xyz coordinates of the tube around the axis, and A is 
# surface.dgamma_by_dcoeff()
s = SurfaceXYZFourier(mpol, ntor, nfp, stellsym, phis, thetas)
s.xc[0, ntor + 1] = 1.
s.xc[1, ntor + 1] = 0.1
s.ys[0, ntor + 1] = 1.
s.ys[1, ntor + 1] = 0.1
s.zs[1, ntor] = 0.1

s.fit_to_curve(ma, 0.1)



M = s.dgamma_by_dcoeff()
M = M.reshape((nphi*ntheta*3, M.shape[3]))
print(np.linalg.matrix_rank(M))
print(M.shape)



dofs = s.get_dofs()
eps = 1e-4
h = np.random.uniform(size=dofs.shape)
r0, J0, Jiota = boozer_surface_residual(s, iota, bs)
print(f"J0.shape={J0.shape}, Rank(J0)={np.linalg.matrix_rank(J0)}")
# J = np.delete(J0, 2*J0.shape[1]//3, 1)
# J = np.concatenate((J, Jiota), axis=1)
# print("Rank(J)", np.linalg.matrix_rank(J))

print("Taylor test the residual")
Jex = J0@h
for eps in [1e-3, 1e-4, 1e-5, 1e-6]:
    s.set_dofs(dofs + eps*h)
    r1, J1, _ = boozer_surface_residual(s, iota, bs)
    Jfd = (r1-r0)/eps
    print(np.linalg.norm(Jfd-Jex)/np.linalg.norm(Jex))

s.set_dofs(dofs)
x = np.concatenate((s.get_dofs(), [iota]))
area0 = s.surface_area()

# Should actually enfore the surface area constraint exactly, but currently
# just adding it as a penalty.
def f(x):
    sdofs = x[:-1]
    iota = x[-1]
    s.set_dofs(sdofs)
    r, Js, Jiota = boozer_surface_residual(s, iota, bs)
    J = np.concatenate((Js, Jiota), axis=1)
    val = 0.5 * np.sum(r**2) + 0.5 * (s.surface_area()-area0)**2
    dval = np.sum(r[:, None]*J, axis=0) + (s.surface_area()-area0)*np.concatenate((s.dsurface_area_by_dcoeff(), [0.]))
    print(val, np.linalg.norm(dval))
    return val, dval

print('Taylor rest 0.5 * residual^2')
f0, J0 = f(x)
h = np.random.uniform(size=x.shape)
Jex = J0@h
for eps in [1e-3, 1e-4, 1e-5, 1e-6]:
    f1, J1 = f(x + eps*h)
    Jfd = (f1-f0)/eps
    print(np.linalg.norm(Jfd-Jex)/np.linalg.norm(Jex))


# def g(x):
#     return , 

s.set_dofs(x[:-1])
xyz0 = s.gamma()
bs.set_points(xyz0.reshape((nphi*ntheta, 3)))
absB0 = np.linalg.norm(bs.B(), axis=1).reshape((nphi, ntheta))
s.plot(scalars=absB0)

# v, dv = f(x)
# for i in range(1000):
#     x -= 1e-4 * dv
#     v, dv = f(x)
#     print(i, v, np.linalg.norm(dv))

# c, dc = g(x)
from scipy.optimize import minimize
res = minimize(f, x, jac=True, method='L-BFGS-B', options={'maxiter': 100})

s.set_dofs(res.x[:-1])
xyz = s.gamma()
bs.set_points(xyz.reshape((nphi*ntheta, 3)))
absB = np.linalg.norm(bs.B(), axis=1).reshape((nphi, ntheta))
s.plot(scalars=absB)


import matplotlib.pyplot as plt
for i in range(ntheta):
    plt.plot(phis, absB[:,i])
    plt.plot(phis, absB0[:,i], ':')
plt.show()


import IPython; IPython.embed()
import sys; sys.exit()
