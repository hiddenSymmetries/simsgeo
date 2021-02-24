import numpy as np
from pyplasmaopt import *
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



def boozer(i, surface, coilCollection, bs, target):            
    xyz = surface.get_dofs()[:-1].reshape( (-1,3) )
    G = 2. * np.pi * jnp.sum( jnp.array( coilCollection._base_currents ) ) * 2. * surface.nfp * (4 * np.pi * 10**(-7) / (2 * np.pi))

    bs.set_points(xyz)
    bs.compute(bs.points)
    B = bs.B.flatten()
    Bmag2     = jnp.tile( (bs.B[:,0]**2 + bs.B[:,1]**2 + bs.B[:,2]**2)[:,None], (1, 3) ).flatten()

    pde     = B - (Bmag2/G) * ( surface.gammadash1().flatten() + i * surface.gammadash2().flatten() )

    if surface.constraint == 'tf':
        cons = jnp.array( [surface.toroidal_flux() - target] )
    elif surface.constraint == 'sa':
        cons = jnp.array( [surface.surface_area() - target] )
    else:
        raise("Constraint not implemented yet!")
    rhs = jnp.concatenate( (pde, cons) )

    didx = jnp.arange(xyz.shape[0]) 
    dB_dX = jnp.zeros( (xyz.shape[0], 3, xyz.shape[0], 3) )
    dB_dX = index_update( dB_dX, index[didx, :,didx , :], np.transpose(bs.dB_by_dX,axes=(0,2,1)) ).reshape( (3 * xyz.shape[0], 3 * xyz.shape[0] ) ) 

    

    dBmag2_dX_lin = 2.*(bs.B[:,0][:,None] * bs.dB_by_dX[:,:,0] \
                      + bs.B[:,1][:,None] * bs.dB_by_dX[:,:,1] \
                      + bs.B[:,2][:,None] * bs.dB_by_dX[:,:,2]).reshape( (-1,3) )
    dBmag2_dX = jnp.zeros( (xyz.shape[0], 3, xyz.shape[0], 3) )
    dBmag2_dX = index_update( dBmag2_dX, index[didx,0,didx,:], dBmag2_dX_lin ) 
    dBmag2_dX = index_update( dBmag2_dX, index[didx,1,didx,:], dBmag2_dX_lin ) 
    dBmag2_dX = index_update( dBmag2_dX, index[didx,2,didx,:], dBmag2_dX_lin ) 
    dBmag2_dX = dBmag2_dX.reshape(  (3 * xyz.shape[0], 3 * xyz.shape[0] ) )

    term1 =  (dBmag2_dX/G) * ( surface.gammadash1() + i * surface.gammadash2() ).reshape( (-1,1) )
    term2 = (Bmag2  / G ).reshape( (-1,1) ) * ( surface.Dphi + i * surface.Dtheta )
    dpde_dX = dB_dX - term1 - term2 
    dpde_di = (  -(Bmag2/G) * surface.gammadash2().flatten() ).reshape( (-1,1) )
    
    if surface.constraint == 'tf':
        cons_dX = surface.toroidal_flux_dx().reshape( (1,-1) )
    elif surface.constraint == 'sa':
        cons_dX = surface.surface_area_dX( xyz ).reshape( (1,-1) )
    else:
        raise("Constraint not implemented yet.")
    cons_di = np.array([[0.]])

    drhs_pde = jnp.hstack( (dpde_dX, dpde_di) )
    drhs_cons = jnp.hstack( (cons_dX, cons_di) )
    drhs = jnp.vstack( (drhs_pde, drhs_cons) )

    rhs = index_update(rhs, index[0], rhs[0] + rhs[1] + rhs[2] )
    drhs = index_update(drhs, index[0,:], drhs[0,:] + drhs[1,:] + drhs[2,:] )
    
    keep = np.concatenate( (np.array([0]), np.arange(3,rhs.size) ) )
    rhs = rhs[keep]
    drhs = drhs[keep,:]
    drhs = drhs[:,keep]
    

    return rhs,drhs


class JaxCartesianMagneticSurface(JaxCartesianSurface):
    def __init__(self, *args, label_target = None, constraint = 'tf'):
        if type(args[0]) is JaxCartesianSurface :
            self.__dict__ = args[0].__dict__.copy()
            self.bs = args[1]
            self.cc = args[2]
            self.iota = args[3]
            self.label_target = label_target
            self.constraint = constraint
            sgpp.Surface.__init__(self, args[0].quadpoints_phi, args[0].quadpoints_theta)
        else:
            quadpoints_phi, quadpoints_theta, nfp, ss, flip, bs, cc = args
            super().__init__(quadpoints_phi, quadpoints_theta, nfp, ss, flip)
            self.bs = bs
            self.cc = cc
            self.constraint = constraint
            self.label_target = None
            if label_target is not None:
                self.label_target = label_target
        
# this one works for all phi = const profiles 
#    def toroidal_flux(self):
#        points = self.apply_symmetries( self.gamma() )
#        self.bs.set_points(points.reshape( (-1,3) ))
#        A = self.bs.A.reshape( (points.shape[0], points.shape[1] , 3) )
#        
#        gammadash2 = np.zeros( points.shape )
#        gammadash2[:,:,0] = points[:,:,0] @ self.D2.T
#        gammadash2[:,:,1] = points[:,:,1] @ self.D2.T
#        gammadash2[:,:,2] = points[:,:,2] @ self.D2.T
#        
#        ipdb.set_trace()
#        profile = np.mean( np.sum( A * gammadash2, axis = -1), axis = -1 )
#        
#        tf = profile[0] # take the first one..., the derivative simplifies if you only take this one I think
#        return tf



    def toroidal_flux(self):
        points = self.gamma()[0,:,:] # take the first profile -- easiest for both ss and non ss surfaces
        Ablock = self.bs.A.reshape( self.gamma().shape )
        A = Ablock[0,:,:]


        gammadash2 = (self.Dtheta1D @ points.flatten()).reshape( (-1,3) )
        gammadash2_x = gammadash2[:,0] 
        gammadash2_y = gammadash2[:,1] 
        gammadash2_z = gammadash2[:,2] 


        dot =  A[:,0] * gammadash2_x + A[:,1] * gammadash2_y + A[:,2] * gammadash2_z
        if self.ss == 1:
            dot = np.concatenate( (dot, dot[1:] ) )

        tf = np.mean(dot)
        return tf


    def toroidal_flux_dx(self):
        points = self.gamma()[0,:,:] 
        Ablock = self.bs.A.reshape( self.gamma().shape )
        A = Ablock[0,:,:]
        dA_by_dX_block = self.bs.dA_by_dX.reshape( (self.numquadpoints_phi, self.numquadpoints_theta,3,3) )
        dA_by_dX = dA_by_dX_block[0,:,:,:]


        didx = jnp.arange(self.numquadpoints_theta) 
        dA_dX = jnp.zeros( (didx.size, 3, didx.size, 3) )
        dA_dX = index_update( dA_dX, index[didx, :,didx , :], np.transpose(dA_by_dX,axes=(0,2,1) ) ).reshape( (3 * didx.size, 3 * didx.size ) ) 

        gammadash2 = self.Dtheta1D @ points.flatten()
        temp = dA_dX * gammadash2.reshape( (-1,1) ) + self.Dtheta1D * A.reshape( (-1,1) )
        dot_dx = np.sum( temp.reshape( (points.shape[0],3,-1) ), axis = -2) 

        if self.ss == 1:
            dot_dx = np.concatenate( (dot_dx, dot_dx[1:,:] ), axis = 0 )
        
        tf_dx_partial = np.mean(dot_dx, axis = 0)
        tf_dx = np.zeros( self.gamma().shape )
        tf_dx[0,:,:] = tf_dx_partial.reshape( (-1,3) )
        tf_dx = tf_dx.flatten()
        return tf_dx

       




    def convert2Boozer(self, xyzi):
        def func(in_xyzi, surface, coilCollection, bs, label_target):
            surface.set_dofs( in_xyzi, False )
            i = in_xyzi[-1]
            f,df = boozer( i, surface, coilCollection, bs, label_target)
            return f,df
        
        lbs = BiotSavart(self.cc.coils, self.cc.currents)
        # create temporary surface
        surf = JaxCartesianMagneticSurface( self.quadpoints_phi, self.quadpoints_theta , self.nfp, self.ss, self.flip, lbs, self.cc, constraint = self.constraint)
        surf.set_dofs(xyzi, False)
        

        if self.label_target is None:
            if self.constraint == "tf":
                self.label_target = surf.toroidal_flux()
            elif self.constraint == "sa":
                self.label_target = surf.surface_area()
            else:
                raise("This constraint is not implemented yet")
            surf.label_target = self.label_target

        fdf = lambda x : func(x, surf, self.cc, self.bs, self.label_target)
        diff = 1
        count = 0
        lamb = 1e-6
        rhs,drhs = fdf(xyzi)
        norm_res = np.linalg.norm(rhs)
        print("initial norm is ", norm_res )
        
        # levenberg marquart is more robust than plain vanilla Newton
        while norm_res > 1e-12:
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
    
    def updateBoozer(self):
        xyzi = self.get_dofs()
        xyzi = self.convert2Boozer(xyzi)
        super().set_dofs(xyzi[:-1])
        self.iota = xyzi[-1]
        self.invalidate_cache()


    def get_dofs(self):
        xyz = super().get_dofs()
        return np.concatenate( (xyz,np.array([self.iota]) ) )

    def set_dofs(self, xyzi, update = True):
        if update:
            xyzi = self.convert2Boozer(xyzi)
        
        super().set_dofs(xyzi[:-1])
        self.iota = xyzi[-1]
        self.bs.set_points( xyzi[:-1].reshape( (-1,3) ) )
        
        self.invalidate_cache()

    def print_metadata(self):
        print("*************************")
        print("Iota : {:.8f}".format( self.iota ) )
        print("Toroidal flux: {:.8f}".format( self.toroidal_flux() ) )
        super().print_metadata()
