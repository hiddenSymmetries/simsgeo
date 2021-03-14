import numpy as np
import ipdb
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
    dB_by_dX    = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
    d2B_by_dXdX = biotsavart.d2B_by_dXdX().reshape((nphi,ntheta,3,3,3))
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
   
#    ipdb.set_trace()
#    d2B_dcdc = np.einsum('ijkpl,ijpn,ijkm->ijlmn', d2B_by_dXdX, dx_dc, dx_dc)
#    dB2_dc = 2.* np.einsum('ijl,ijlm->ijm', B, dB_dc)
#    
#    term1 = np.einsum('ijlm,ijln->ijmn',dB_dc, dB_dc)
#    term2 = np.einsum('ijlmn,ijl->ijmn',d2B_dcdc,B)
#    d2B2_dcdc = 2*(term1 + term2) 
#
#    term1 = -(1/G) * ( dxphi_dc[...,None,:] - iota * dxtheta_dc[...,None,:] ) * dB2_dc[...,None,:,None]
#    term2 = -(1/G) * ( dxphi_dc[...,:,None] - iota * dxtheta_dc[...,:,None] ) * dB2_dc[...,None,None,:]
#    term3 = -(1/G) * (  xphi[...,None,None] - iota * xtheta[...,None,None]  ) * d2B2_dcdc[...,None,:,:]
#    d2residual_by_dcdc = d2B_dcdc + term1 + term2 + term3
    
    



    residual_flattened = residual.reshape((nphi*ntheta*3, ))
    dresidual_dc_flattened = dresidual_dc.reshape((nphi*ntheta*3, dresidual_dc.shape[-1]))
    dresidual_diota_flattened = dresidual_diota.reshape((nphi*ntheta*3, 1))

    return residual_flattened, dresidual_dc_flattened, dresidual_diota_flattened

def minimum_flux_surface_residual(surface, iota, biotsavart):
    pass

class ToroidalFlux(object):

    r"""
    This objective calculates
        J = \int_{varphi = constant} B \cdot n ds
          = \int_{varphi = constant} curlA \cdot n ds
          from Stokes' theorem
          = \int_{curve on surface where varphi = constant} A \cdot n dl
    given a surface and Biot Savart kernel.
    """

    def __init__(self, surface, biotsavart):
        self.surface = surface
        self.biotsavart = biotsavart
        self.idx = 0 # varphi = 0 here

    def J(self):
        x = self.surface.gamma()[self.idx]
        xtheta = self.surface.gammadash2()[self.idx]
        ntheta = self.surface.gamma().shape[1]
    
        self.biotsavart.set_points(x)
        A = self.biotsavart.A()

        tf = np.sum(A * xtheta)/ntheta
        return tf

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the derivatives with respect to the surface coefficients
        """
        ntheta = self.surface.gamma().shape[1]
        x = self.surface.gamma()[self.idx]
        self.biotsavart.set_points(x)
 
        A = self.biotsavart.A()
        dA_by_dX = self.biotsavart.dA_by_dX()
        dgammadash2 = self.surface.gammadash2()[self.idx,:]
        dgammadash2_by_dc = self.surface.dgammadash2_by_dcoeff()[self.idx,:]

        dx_dc = self.surface.dgamma_by_dcoeff()[self.idx]
        dA_dc = np.sum( dA_by_dX[...,:,None] * dx_dc[...,None,:],axis=1)
        term1 = np.sum( dA_dc * dgammadash2[...,None], axis = (0,1) ) 
        term2 = np.sum( A[...,None] * dgammadash2_by_dc, axis = (0,1) )

        out = (term1+term2)/ntheta
        return out
    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second derivatives with respect to the surface coefficients
        """
        x = self.surface.gamma()[self.idx]
        self.biotsavart.set_points(x)
 

        ntheta = self.surface.gamma().shape[1]
        dx_dc = self.surface.dgamma_by_dcoeff()[self.idx]
        dA_by_dX = self.biotsavart.dA_by_dX()
        d2A_by_dXdX = self.biotsavart.d2A_by_dXdX().reshape((ntheta,3,3,3))
        dA_dc = np.sum(dA_by_dX[...,:,None] * dx_dc[...,None,:],axis=1)
        d2A_dcdc = np.einsum('jkpl,jpn,jkm->jlmn', d2A_by_dXdX, dx_dc, dx_dc)

        dgammadash2 = self.surface.gammadash2()[self.idx]
        dgammadash2_by_dc = self.surface.dgammadash2_by_dcoeff()[self.idx]

        term1 = np.sum(d2A_dcdc * dgammadash2[...,None,None],axis = -3)
        term2 = np.sum(dA_dc[...,:,None] * dgammadash2_by_dc[...,None,:], axis = -3)
        term3 = np.sum(dA_dc[...,None,:] * dgammadash2_by_dc[...,:,None], axis = -3)
        
        out = (1/ntheta) * np.sum(term1+term2+term3, axis = 0)
        return out



#        A = self.biotsavart.A()
#        dA_by_dX = self.biotsavart.dA_by_dX()
#        dX_by_dcoeff = self.surface.dgamma_by_dcoeff()[self.idx,:]
#        dgammadash2 = self.surface.gammadash2()[self.idx,:]
#        dgammadash2_by_dcoeff = self.surface.dgammadash2_by_dcoeff()[self.idx,:]
        
#        term1 = np.einsum('ijk,ijl,ik->l', dA_by_dX, dX_by_dcoeff, dgammadash2)
#        term2 = np.einsum('ij,ijk->k', A, dgammadash2_by_dcoeff)
#        term2 = np.sum( A[...,None] * dgammadash2_by_dcoeff, axis = (0,1) )


        #dA_dc = np.einsum('jkl,jkm->jlm', dA_by_dX, dx_dc)
