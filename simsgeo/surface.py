import numpy as np
import simsgeopp as sgpp
from jax import grad, vjp, jacfwd, jvp
from .jit import jit
import jax.numpy as jnp
import ipdb
from jax.ops import index, index_update
from jax.numpy.linalg import norm
import scipy.linalg as sp

class Surface():
    def __init__(self, quadpoints_phi, quadpoints_theta):
        self.dependencies = []
        self.phi_grid, self.theta_grid = np.meshgrid(quadpoints_phi, quadpoints_theta)
        self.phi_grid = self.phi_grid.T    
        self.theta_grid = self.theta_grid.T 

    def plot(self, ax=None, show=True, plot_derivative=False, closed_loop=False, color=None, linestyle=None, apply_symmetries = True):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    
       # ipdb.set_trace(context=21)
        gamma = self.gamma()
        dgamma1 = self.gammadash1()
        dgamma2 = self.gammadash2()
        normal = self.normal()

        if apply_symmetries:
            gamma = self.apply_symmetries( gamma )
            dgamma1 = self.apply_symmetries( dgamma1 )
            dgamma2 = self.apply_symmetries( dgamma2 )
            normal = self.apply_symmetries( normal )
        def rep(data):
            if closed_loop:
                concat_theta = np.vstack((data, data[0,:][None,:]))
                data = np.hstack((concat_theta, concat_theta[:,0][:,None]))
                return data
            else:
                return data
        
        from mayavi import mlab
        mlab.mesh(rep(gamma[:,:,0]), rep(gamma[:,:,1]), rep(gamma[:,:,2]))
        mlab.mesh(rep(gamma[:,:,0]), rep(gamma[:,:,1]), rep(gamma[:,:,2]), representation='wireframe', color = (0,0,0))
        

        if plot_derivative:
            mlab.quiver3d(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2],\
                      0.05*dgamma1[:,:,0], 0.05*dgamma1[:,:,1], 0.05*dgamma1[:,:,2])
            mlab.quiver3d(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2],\
                      0.05*dgamma2[:,:,0], 0.05*dgamma2[:,:,1], 0.05*dgamma2[:,:,2])
#            mlab.quiver3d(gamma[:,:,0], gamma[:,:,1], gamma[:,:,2],\
#                      0.005*normal[:,:,0], 0.005*normal[:,:,1], 0.005*normal[:,:,2])
        mlab.show()        
      


    def apply_symmetries(self, gamma):
        return gamma

class JaxSurface(sgpp.Surface, Surface):
    def __init__(self, quadpoints_phi, quadpoints_theta, pure):
        if isinstance(quadpoints_phi, np.ndarray):
            quadpoints_phi = list(quadpoints_phi)
            quadpoints_theta = list(quadpoints_theta)
        sgpp.Surface.__init__(self, quadpoints_phi, quadpoints_theta)
        Surface.__init__(self, quadpoints_phi, quadpoints_theta)
        self.gamma_pure = pure
        self.gamma_jax = jit(lambda dofs: self.gamma_pure(dofs, quadpoints_phi, quadpoints_theta  ) )
        
        # implement the differentiation matrices

    def gamma_impl(self, gamma):
        dofs = self.get_dofs()[:3 * self.numquadpoints_phi * self.numquadpoints_theta]
        gamma[:,:,:] = self.gamma_jax(dofs)
    def gammadash1_impl(self, dgamma_dphi):
        dgamma_dphi[:,:,:] = (self.Dphi @ self.gamma().flatten()   ).reshape( self.gamma().shape ) 
    def gammadash2_impl(self, dgamma_dtheta):
        dgamma_dtheta[:,:,:] = (self.Dtheta @ self.gamma().flatten() ).reshape( self.gamma().shape ) 
    def normal_impl(self, normal):
        normal[:,:,:] = jnp.cross(self.gammadash1(), self.gammadash2() )
    def surface_area(self):
        n = self.normal()
        return jnp.mean( jnp.sqrt(n[:,:,0]**2 + n[:,:,1]**2 + n[:,:,2]**2 ) )
        
def cartesianboozersurface_pure(dofs, quadpoints_phi, quadpoints_theta):
    gamma =  jnp.array( dofs.reshape( (len(quadpoints_phi), len(quadpoints_theta),3) ) )
    return gamma

class JaxCartesianSurface(JaxSurface):
    def __init__(self, quadpoints_phi, quadpoints_theta, nfp, ss,flip):
        pure = lambda dofs, quadpoints_phi, quadpoints_theta: cartesianboozersurface_pure(dofs,quadpoints_phi, quadpoints_theta)
        super().__init__(quadpoints_phi, quadpoints_theta, pure)
        self.numquadpoints_phi = len(quadpoints_phi)
        self.numquadpoints_theta = len(quadpoints_theta)
        self.spatial_size = len(quadpoints_phi) * len(quadpoints_theta) 
        self.coefficients = [np.zeros((self.numquadpoints_phi,self.numquadpoints_theta)), 
                             np.zeros((self.numquadpoints_phi,self.numquadpoints_theta)),
                             np.zeros((self.numquadpoints_phi,self.numquadpoints_theta)) ]
        
        self.surface_area_dX = grad( self.surface_area_pure )

        self.nfp = nfp
        self.flip = flip
        self.ss = ss
        def generate_diff_matrix(N, xMin, xMax):
            h = 2 * np.pi / N   #grid spacing
            kk = np.linspace(1, N - 1, N - 1)
            
            n1 = int(np.floor((N - 1)/2))
            n2 = int(np.ceil((N - 1)/2))
            
            if N % 2 == 0:
                topc = 1 / (np.tan(np.linspace(1, n2, n2) * (h/2)))
                col1 = np.concatenate((np.array([0]), 0.5 * np.power(-1, kk) * np.concatenate((topc, -np.flip(topc[0:n1])))))
                col1[int(N/2)] = 0 #force this element to zero to avoid weird floating point issues
            else:
                topc = 1 / (np.sin(np.linspace(1, n2, n2) * (h/2)))
                col1 = np.concatenate((np.array([0]), 0.5 * np.power(-1, kk) * np.concatenate((topc, np.flip(topc[0:n1])))))
            row1 = -col1
            D = (2 * np.pi)/(xMax - xMin) * sp.toeplitz(col1, row1)
            return D
        
        self.D1 = jnp.array( generate_diff_matrix( self.numquadpoints_phi * self.nfp , 0, 1) )
        if self.ss == 1:
            self.D2 = jnp.array( generate_diff_matrix( self.numquadpoints_theta * 2 - 1,      0, 1) )
        else:
            self.D2 = jnp.array( generate_diff_matrix( self.numquadpoints_theta ,      0, 1) )


        def dgamma_dphi(in_gamma):
           
            in_gamma_reshaped = in_gamma.reshape( (self.numquadpoints_phi, self.numquadpoints_theta,3) )

            if self.ss == 1:
                gamma_temp = jnp.zeros( (self.nfp * self.numquadpoints_phi, 2 * self.numquadpoints_theta-1, 3) )
            else:
                gamma_temp = jnp.zeros( (self.nfp * self.numquadpoints_phi,  self.numquadpoints_theta, 3) )

            gamma_out = jnp.zeros( (self.numquadpoints_phi, self.numquadpoints_theta,3 ) )

            gamma_sym = self.apply_symmetries( in_gamma_reshaped )
            gamma_temp = index_update(gamma_temp,index[:,:,0], self.D1 @ gamma_sym[:,:,0] ) 
            gamma_temp = index_update(gamma_temp,index[:,:,1], self.D1 @ gamma_sym[:,:,1] ) 
            gamma_temp = index_update(gamma_temp,index[:,:,2], self.D1 @ gamma_sym[:,:,2] ) 

            gamma_out = index_update(gamma_out,index[:,:,0], gamma_temp[:self.numquadpoints_phi, :self.numquadpoints_theta, 0] ) 
            gamma_out = index_update(gamma_out,index[:,:,1], gamma_temp[:self.numquadpoints_phi, :self.numquadpoints_theta, 1] ) 
            gamma_out = index_update(gamma_out,index[:,:,2], gamma_temp[:self.numquadpoints_phi, :self.numquadpoints_theta, 2] ) 
            return gamma_out.flatten()

        def dgamma_dtheta(in_gamma):
            in_gamma_reshaped = in_gamma.reshape( (self.numquadpoints_phi, self.numquadpoints_theta,3) )
            
            if self.ss == 1:
                gamma_temp = jnp.zeros( (self.nfp * self.numquadpoints_phi, 2 * self.numquadpoints_theta-1, 3) )
            else:
                gamma_temp = jnp.zeros( (self.nfp * self.numquadpoints_phi,  self.numquadpoints_theta, 3) )
                
            gamma_out = jnp.zeros( (self.numquadpoints_phi, self.numquadpoints_theta,3 ) )
            
            gamma_sym = self.apply_symmetries( in_gamma_reshaped )
            gamma_temp = index_update(gamma_temp,index[:,:,0], gamma_sym[:,:,0] @ self.D2.T ) 
            gamma_temp = index_update(gamma_temp,index[:,:,1], gamma_sym[:,:,1] @ self.D2.T ) 
            gamma_temp = index_update(gamma_temp,index[:,:,2], gamma_sym[:,:,2] @ self.D2.T ) 
            
            gamma_out = index_update(gamma_out,index[:,:,0], gamma_temp[:self.numquadpoints_phi, :self.numquadpoints_theta, 0] ) 
            gamma_out = index_update(gamma_out,index[:,:,1], gamma_temp[:self.numquadpoints_phi, :self.numquadpoints_theta, 1] ) 
            gamma_out = index_update(gamma_out,index[:,:,2], gamma_temp[:self.numquadpoints_phi, :self.numquadpoints_theta, 2] ) 
            return gamma_out.flatten()

        def dgamma_dtheta_1D(in_gamma):
            in_gamma_reshaped = in_gamma.reshape( ( self.numquadpoints_theta,3) )
            
            if self.ss == 1:
                gamma_temp = jnp.zeros( ( 2 * self.numquadpoints_theta-1, 3) )
                S = jnp.array([ [1, 0, 0],[0, -1, 0], [0,0,-1] ])
                in_gamma_sym = in_gamma_reshaped @ S.T
                gamma_sym = jnp.concatenate( (in_gamma_reshaped, jnp.flipud( in_gamma_sym[1:,:] ) ), axis = 0)
            else:
                gamma_temp = jnp.zeros( (self.numquadpoints_theta, 3) )
                gamma_sym = in_gamma_reshaped
                
            gamma_out = jnp.zeros( ( self.numquadpoints_theta,3 ) )
            
            #gamma_sym = self.apply_symmetries( in_gamma_reshaped )
            gamma_temp = index_update(gamma_temp,index[:,0], gamma_sym[:,0] @ self.D2.T ) 
            gamma_temp = index_update(gamma_temp,index[:,1], gamma_sym[:,1] @ self.D2.T ) 
            gamma_temp = index_update(gamma_temp,index[:,2], gamma_sym[:,2] @ self.D2.T ) 
            
            gamma_out = index_update(gamma_out,index[:,0], gamma_temp[:self.numquadpoints_theta, 0] ) 
            gamma_out = index_update(gamma_out,index[:,1], gamma_temp[:self.numquadpoints_theta, 1] ) 
            gamma_out = index_update(gamma_out,index[:,2], gamma_temp[:self.numquadpoints_theta, 2] ) 
            return gamma_out.flatten()
        
        self.Dphi =   jacfwd( lambda g: dgamma_dphi(g)    )( np.zeros( (3 * self.numquadpoints_phi * self.numquadpoints_theta, ) ) ) 
        self.Dtheta = jacfwd( lambda g: dgamma_dtheta(g)  )( np.zeros( (3 * self.numquadpoints_phi * self.numquadpoints_theta, ) ) ) 
        self.Dtheta1D = jacfwd( lambda g: dgamma_dtheta_1D(g)  )( np.zeros( (3 * self.numquadpoints_theta, ) ) ) 
         


    def apply_symmetries(self,gamma):
        nfp = self.nfp
        flip = self.flip
        ss = self.ss


        X = gamma[:,:,0]
        Y = gamma[:,:,1]
        Z = gamma[:,:,2]

        S = jnp.array([ [1, 0, 0],[0, -1, 0], [0,0,-1] ])
        xyz = jnp.vstack( (X.reshape( (1,-1) ), Y.reshape( (1,-1) ), Z.reshape( (1,-1) ) ) )
        
        XX = jnp.array(X)
        YY = jnp.array(Y)
        ZZ = jnp.array(Z)
        
        for i in range(1,nfp):
            R = jnp.array([ [ np.cos(flip * i * 2. * np.pi / nfp), np.sin(flip * i * 2. * np.pi / nfp ), 0.], \
                            [-np.sin(flip * i * 2. * np.pi / nfp), np.cos(flip * i * 2. * np.pi / nfp ), 0.], \
                           [ 0, 0, 1.] ] )
            rotated = R @ xyz
            XX = jnp.vstack( (XX, rotated[0,:].reshape( X.shape ) ) )
            YY = jnp.vstack( (YY, rotated[1,:].reshape( Y.shape ) ) )
            ZZ = jnp.vstack( (ZZ, rotated[2,:].reshape( Z.shape ) ) )
        
        if self.ss == 1:
            xyz = jnp.vstack( (XX.reshape( (1,-1) ), YY.reshape( (1,-1) ), ZZ.reshape( (1,-1) ) ) )
            xyz_ref = (S @ xyz)

            xyz_ss = jnp.zeros( (nfp * self.numquadpoints_phi, self.numquadpoints_theta, 3) )
            xyz_ss = index_update(xyz_ss, index[:,:,0], xyz_ref[0,:].reshape( (nfp * self.numquadpoints_phi, self.numquadpoints_theta) ) )
            xyz_ss = index_update(xyz_ss, index[:,:,1], xyz_ref[1,:].reshape( (nfp * self.numquadpoints_phi, self.numquadpoints_theta) ) )
            xyz_ss = index_update(xyz_ss, index[:,:,2], xyz_ref[2,:].reshape( (nfp * self.numquadpoints_phi, self.numquadpoints_theta) ) )
            
            v1 = jnp.roll(jnp.fliplr(jnp.flipud(xyz_ss[:,1:, 0]  ) ), 1, axis = 0) 
            v2 = jnp.roll(jnp.fliplr(jnp.flipud(xyz_ss[:,1:, 1]  ) ), 1, axis = 0)
            v3 = jnp.roll(jnp.fliplr(jnp.flipud(xyz_ss[:,1:, 2]  ) ), 1, axis = 0)
            
            XX = jnp.hstack( (XX, v1 )) 
            YY = jnp.hstack( (YY, v2 ))
            ZZ = jnp.hstack( (ZZ, v3 ))
        
        out_gamma = jnp.zeros( (XX.shape[0], XX.shape[1],3) )
        
        out_gamma = index_update(out_gamma,index[:,:,0], XX ) 
        out_gamma = index_update(out_gamma,index[:,:,1], YY ) 
        out_gamma = index_update(out_gamma,index[:,:,2], ZZ ) 

        return out_gamma 

    def num_dofs(self):
        return 3*self.numquadpoints_phi * self.numquadpoints_theta

    def get_dofs(self):
        return np.concatenate([coeffs.flatten() for coeffs in self.coefficients])

    def set_dofs_impl(self, dofs):
        dofs = np.array(dofs)
        self.coefficients[0][:] = dofs[:self.spatial_size].reshape( (self.numquadpoints_phi, self.numquadpoints_theta) )
        self.coefficients[1][:] = dofs[self.spatial_size:2*self.spatial_size].reshape( (self.numquadpoints_phi, self.numquadpoints_theta) )
        self.coefficients[2][:] = dofs[2*self.spatial_size:].reshape( (self.numquadpoints_phi, self.numquadpoints_theta) )

        # convert the dofs to Boozer coordinates
        self.invalidate_cache()
    def surface_area_pure(self, in_g ):
        t1 = (self.Dphi @ in_g.flatten()   ).reshape( in_g.shape )
        t2 = (self.Dtheta @ in_g.flatten()   ).reshape( in_g.shape )
        n = jnp.cross( t1, t2 )
        return jnp.mean( jnp.sqrt(n[:,0]**2 + n[:,1]**2 + n[:,2]**2 ) )

    def interpolated_surface(self, Nphi_new, Ntheta_new):
        gg = self.apply_symmetries(self.gamma())
        GG_x = np.fft.fft2( gg[:,:,0] )
        GG_y = np.fft.fft2( gg[:,:,1] )
        GG_z = np.fft.fft2( gg[:,:,2] )
       
#        ipdb.set_trace(context=21)
        GG_x = np.fft.fftshift(GG_x) 
        GG_y = np.fft.fftshift(GG_y) 
        GG_z = np.fft.fftshift(GG_z) 

        # in order to always have an odd dimensioned differentiation matrix, this should be edited for when nfp is even
        dphi   = int((self.nfp * Nphi_new   - gg.shape[0])/2  )
        dtheta = int( (2 * Ntheta_new-1 - gg.shape[1])/2 )
        
       
        GG_x = np.pad( GG_x, ((dphi, dphi), (dtheta, dtheta)), 'constant', constant_values = (0) )
        GG_y = np.pad( GG_y, ((dphi, dphi), (dtheta, dtheta)), 'constant', constant_values = (0) )
        GG_z = np.pad( GG_z, ((dphi, dphi), (dtheta, dtheta)), 'constant', constant_values = (0) )
        
        GG_x = np.fft.ifftshift(GG_x)
        GG_y = np.fft.ifftshift(GG_y)
        GG_z = np.fft.ifftshift(GG_z)

#        ipdb.set_trace(context=21)
        gg_x = np.real(np.fft.ifft2(GG_x) ) * (Nphi_new * Ntheta_new)/(self.numquadpoints_phi * self.numquadpoints_theta)
        gg_y = np.real(np.fft.ifft2(GG_y) ) * (Nphi_new * Ntheta_new)/(self.numquadpoints_phi * self.numquadpoints_theta)
        gg_z = np.real(np.fft.ifft2(GG_z) ) * (Nphi_new * Ntheta_new)/(self.numquadpoints_phi * self.numquadpoints_theta)
        
        Nphi_updated   = int(gg_x.shape[0]/self.nfp) 
        Ntheta_updated = int(1 + gg_x.shape[1]/2 )
        phi   = np.linspace(0, 1./self.nfp,       Nphi_updated, endpoint = False)
        theta = np.linspace(0, 1.       , 2*Ntheta_updated-1,   endpoint = False)
        theta = theta[:Ntheta_updated]
        
        gg = np.zeros( (Nphi_updated, Ntheta_updated, 3) )
        gg[:,:,0] = gg_x[:Nphi_updated, :Ntheta_updated]
        gg[:,:,1] = gg_y[:Nphi_updated, :Ntheta_updated]
        gg[:,:,2] = gg_z[:Nphi_updated, :Ntheta_updated]
        

        surf = JaxCartesianSurface( phi, theta , self.nfp, self.ss, self.flip)
        surf.set_dofs( gg.flatten() )
        return surf












