from simsgeo import StelleratorSymmetricCylindricalFourierCurve, FourierCurve
import numpy as np


""" This curve was implemented in python """
ma = StelleratorSymmetricCylindricalFourierCurve(20, 3, 2)
ma.coefficients[0][0] = 1.
ma.plot(plot_derivative=True)


""" This curve was implemented in C++, but can be used in the same way """
order = 2
cu = FourierCurve(50, order)
dofs = np.zeros((cu.num_dofs(), ))
dofs[1] = 1.
dofs[2*order+3] = 1.
dofs[4*order+3] = 1.
cu.set_dofs(dofs)
cu.plot(plot_derivative=True)
