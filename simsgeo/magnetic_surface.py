import numpy as np
import simsgeopp as sgpp
from jax import grad, vjp, jacfwd, jvp
from .jit import jit
import jax.numpy as jnp
import ipdb
from jax.ops import index, index_update
from jax.numpy.linalg import norm
import scipy.linalg as sp
from simsgeo import JaxStelleratorSymmetricCylindricalFourierCurve, StelleratorSymmetricCylindricalFourierCurve, \
FourierCurve, JaxFourierCurve, RotatedCurve, JaxCartesianSurface, stelleratorsymmetriccylindricalfouriercurve_pure



def boozer(i, surface, coilCollection, bs, sa_target):            
    xyz = surface.get_dofs().reshape( (-1,3) )
    G = 2. * np.pi * jnp.sum( jnp.array( coilCollection._base_currents ) ) * 2. * surface.nfp * (4 * np.pi * 10**(-7) / (2 * np.pi))

    bs.set_points(xyz)
    bs.compute(bs.points)
    B = bs.B.flatten()
    Bmag2     = jnp.tile( (bs.B[:,0]**2 + bs.B[:,1]**2 + bs.B[:,2]**2)[:,None], (1, 3) ).flatten()

    pde     = B - (Bmag2/G) * ( surface.gammadash1().flatten() + i * surface.gammadash2().flatten() )
    sa_cons = jnp.array( [surface.surface_area() - sa_target] )
    rhs = jnp.concatenate( (pde, sa_cons) )

    didx = jnp.arange(xyz.shape[0]) 
    dB_dX = jnp.zeros( (xyz.shape[0], 3, xyz.shape[0], 3) )
    dB_dX = index_update( dB_dX, index[didx, :,didx , :], bs.dB_by_dX ).reshape( (3 * xyz.shape[0], 3 * xyz.shape[0] ) ) 

    

    dBmag2_dX_lin = 2.*(bs.B[:,0][:,None] * bs.dB_by_dX[:,0,:] \
                      + bs.B[:,1][:,None] * bs.dB_by_dX[:,1,:] \
                      + bs.B[:,2][:,None] * bs.dB_by_dX[:,2,:]).reshape( (-1,3) )
    dBmag2_dX = jnp.zeros( (xyz.shape[0], 3, xyz.shape[0], 3) )
    dBmag2_dX = index_update( dBmag2_dX, index[didx,0,didx,:], dBmag2_dX_lin ) 
    dBmag2_dX = index_update( dBmag2_dX, index[didx,1,didx,:], dBmag2_dX_lin ) 
    dBmag2_dX = index_update( dBmag2_dX, index[didx,2,didx,:], dBmag2_dX_lin ) 
    dBmag2_dX = dBmag2_dX.reshape(  (3 * xyz.shape[0], 3 * xyz.shape[0] ) )

    term1 =  (dBmag2_dX/G) * ( surface.gammadash1() + i * surface.gammadash2() ).reshape( (-1,1) )
    term2 = (Bmag2  / G ).reshape( (-1,1) ) * ( surface.Dphi + i * surface.Dtheta )
    dpde_dX = dB_dX - term1 - term2 
    dpde_di = (  -(Bmag2/G) * surface.gammadash2().flatten() ).reshape( (-1,1) )

    sa_cons_dX = surface.surface_area_dX( xyz ).reshape( (1,-1) )
    sa_cons_di = np.array([[0.]])

    drhs_pde = jnp.hstack( (dpde_dX, dpde_di) )
    drhs_cons = jnp.hstack( (sa_cons_dX, sa_cons_di) )
    drhs = jnp.vstack( (drhs_pde, drhs_cons) )

    rhs = index_update(rhs, index[0], rhs[0] + rhs[1] + rhs[2] )
    drhs = index_update(drhs, index[0,:], drhs[0,:] + drhs[1,:] + drhs[2,:] )
    
    keep = np.concatenate( (np.array([0]), np.arange(3,rhs.size) ) )
    rhs = rhs[keep]
    drhs = drhs[keep,:]
    drhs = drhs[:,keep]
    

    return rhs,drhs



class JaxCartesianMagneticSurface(JaxCartesianSurface):
    def __init__(self, quadpoints_phi, quadpoints_theta, nfp, ss, flip, label_target, bs, cc):
        super().__init__(quadpoints_phi, quadpoints_theta, nfp, ss, flip)
        self.bs = bs
        self.cc = cc
        self.label_target = label_target


    def toroidal_flux(self, bs, surf):
        points = surf.get_dofs().reshape( (surf.numquadpoints_phi, surf.numquadpoints_theta,3) )[0,:,:]
        bs.set_points(points)
        A = bs.A
        tf = np.mean(np.sum(A * surf.gammadash2(),axis=1) )
        return tf
    
    def convert2Boozer(self, xyzi):
        def func(in_xyzi, surface, coilCollection, bs, label_target):
            surface.set_dofs( in_xyzi[:-1] )
            i = in_xyzi[-1]
            f,df = boozer( i, surface, coilCollection, bs, label_target)
            return f,df
        
        
        # create temporary surface
        surf = JaxCartesianSurface( self.quadpoints_phi, self.quadpoints_theta , self.nfp, self.ss, self.flip)
        surf.set_dofs(xyzi[:-1])
        
        fdf = lambda x : func(x, surf, self.cc, self.bs, self.label_target)
        diff = 1
        count = 0
        lamb = 1e-6
        rhs,drhs = fdf(xyzi)
        norm_res = np.linalg.norm(rhs)
        print("initial norm is ", norm_res )
        
        # levenberg marquart is more robust than plain vanilla Newton
        while norm_res > 1e-13:
            rhs,drhs = fdf(xyzi)
            
            update = np.linalg.solve(drhs.T @ drhs + lamb * jnp.eye(drhs.shape[0]), drhs.T @ rhs)
            update = np.concatenate( (np.array([update[0],0,0] ), update[1:]) )
            rhstemp,_ = fdf( xyzi -  update)
            while np.linalg.norm(rhstemp) >  np.linalg.norm(rhs):
                lamb = lamb * 10
                update = np.linalg.solve(drhs.T @ drhs + lamb * jnp.eye(drhs.shape[0]), drhs.T @ rhs)
                update = np.concatenate( (np.array([update[0],0,0] ), update[1:]) )
                rhstemp,_ = fdf(xyzi -  update)
            
            lamb = lamb/10
            xyzi = jnp.array(xyzi-update)
            norm_res = np.linalg.norm(rhstemp)
            count += 1
            
            print("norm_res ", norm_res , "lambda ", lamb)
        return xyzi
    
#    def updateBoozer(self):
#        xyzi = self.get_dofs()
#        xyzi = self.convert2Boozer(xyzi)
#        super().set_dofs(xyzi[:-1])
#        self.iota = xyzi[-1]
#        self.invalidate_cache()
#
#
#    def get_dofs(self):
#        xyz = super().get_dofs()
#        return np.concatenate( (xyz,self.iota) )

    def set_dofs(self, xyzi):
        xyzi = self.convert2Boozer(xyzi)

        super().set_dofs(xyzi[:-1])
        self.iota = xyzi[-1]
        
        self.invalidate_cache()

 
