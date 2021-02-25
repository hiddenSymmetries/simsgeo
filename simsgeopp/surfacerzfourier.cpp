#pragma once

#include "surface.cpp"

template<class Array>
class SurfaceRZFourier : public Surface<Array> {
    /*
       SurfaceRZFourier is a surface that is represented in cylindrical
       coordinates using the following Fourier series: 
       r(theta, phi) = \sum_{m=0}^{mpol} \sum_{n=-ntor}^{ntor} [
       r_{c,m,n} \cos(m \theta - n nfp \phi)
       + r_{s,m,n} \sin(m \theta - n nfp \phi) ]
       and the same for z(theta, phi).
       Here, (r, phi, z) are standard cylindrical coordinates, and theta
       is any poloidal angle.
       */

    public:
        using Surface<Array>::quadpoints_phi;
        using Surface<Array>::quadpoints_theta;
        using Surface<Array>::numquadpoints_phi;
        using Surface<Array>::numquadpoints_theta;
        Array rc;
        Array rs;
        Array zc;
        Array zs;
        int nfp;
        int mpol;
        int ntor;

        SurfaceRZFourier(int _mpol, int _ntor, int _nfp, vector<double> _quadpoints_phi, vector<double> _quadpoints_theta)
            : mpol(_mpol), ntor(_ntor), nfp(_nfp), quadpoints_phi(_quadpoints_phi),quadpoints_theta( _quadpoints_theta) {
                numquadpoints_phi = quadpoints_phi.size();
                numquadpoints_theta = quadpoints_theta.size();
                rc = xt::zeros<double>({mpol, 2*ntor+1});
                rs = xt::zeros<double>({mpol, 2*ntor+1});
                zc = xt::zeros<double>({mpol, 2*ntor+1});
                zs = xt::zeros<double>({mpol, 2*ntor+1});
            }

        virtual int num_dofs() = 0;
        virtual void set_dofs_impl(const vector<double>& _dofs) = 0;
        virtual vector<double> get_dofs() = 0;

        void gamma_impl(Array& data) override {
            data *= 0;           
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    double r = 0;
                    double z = 0;
                    for (int i = 0; i < 2*ntor+1; ++i) {
                        for (int m = 0; m < mpol; ++m) {
                            int n  = i - ntor;
                            r += rc(m, i) * cos(m*theta-n*nfp*phi) + rs(m, i) * sin(m*theta-n*nfp*phi);
                            z += zc(m, i) * cos(m*theta-n*nfp*phi) + zs(m, i) * sin(m*theta-n*nfp*phi);
                        }
                    }
                    data(k1, k2, 0) = r * cos(phi);
                    data(k1, k2, 1) = r * sin(phi);
                    data(k1, k2, 2) = z;
                }
            }
        }

        virtual void gammadash1_impl(Array& data) = 0;
        virtual void gammadash2_impl(Array& data) = 0;
        virtual void normal_impl(Array& data) = 0;
        virtual double surface_area() = 0;

};
