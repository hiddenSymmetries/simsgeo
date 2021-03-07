import numpy as np

def boozer_surface_residual(surface, iota, biotsavart):
    """
    For a given surface with points x on it, this function computes the
    residual

        B_BS(x) - (||B_BS(x)||^2/G) * (x_phi - iota * x_theta)

    as well as the derivatives of this residual with respect to surface dofs
    and iota.
    """


    x = surface.gamma()
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    nphi = x.shape[0]
    ntheta = x.shape[1]

    xsemiflat = x.reshape((x.size//3, 3)).copy()

    biotsavart.set_points(xsemiflat)
    dB_by_dX = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
    B = biotsavart.B().reshape((nphi, ntheta, 3))

    tang = xphi + iota * xtheta
    # G = np.sum(np.abs(biotsavart.coil_currents))
    G = 2. * np.pi * np.sum(np.abs(biotsavart.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))
    residual = B - (np.sum(B**2, axis=2)/G)[..., None] * tang

    dx_dc = surface.dgamma_by_dcoeff()
    dxphi_dc = surface.dgammadash1_by_dcoeff()
    dxtheta_dc = surface.dgammadash2_by_dcoeff()


    dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc)

    dresidual_dc =  dB_dc \
        - (2/G) * np.sum(B[..., None]*dB_dc, axis=2)[:, :, None, :] * tang[..., None] \
        - (np.sum(B**2, axis=2)/G)[..., None, None] * (dxphi_dc + iota * dxtheta_dc)
    dresidual_diota = -(np.sum(B**2, axis=2)/G)[..., None] * xtheta

    residual_flattened = residual.reshape((nphi*ntheta*3, ))
    dresidual_dc_flattened = dresidual_dc.reshape((nphi*ntheta*3, dresidual_dc.shape[-1]))
    dresidual_diota_flattened = dresidual_diota.reshape((nphi*ntheta*3, 1))

    return residual_flattened, dresidual_dc_flattened, dresidual_diota_flattened

def minimum_flux_surface_residual(surface, iota, biotsavart):
    pass
