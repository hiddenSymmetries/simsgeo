#pragma once

#include "surface.cpp"

template<class Array>
class SurfaceXYZFourier : public Surface<Array> {
    /*
       SurfaceXYZFourier is a surface that is represented in cartesian
       coordinates using the following Fourier series: 
       x(theta, phi) = \sum_{m=0}^{mpol} \sum_{n=-ntor}^{ntor} [
       x_{c,m,n} \cos(m \theta - n nfp \phi)
       + x_{s,m,n} \sin(m \theta - n nfp \phi) ]
       and the same for y(theta, phi), z(theta, phi).

       Note that for m=0 we skip the n<0 term for the cos terms, and the n<=0
       for the sin terms.
       */

    public:
        using Surface<Array>::quadpoints_phi;
        using Surface<Array>::quadpoints_theta;
        using Surface<Array>::numquadpoints_phi;
        using Surface<Array>::numquadpoints_theta;
        Array xc;
        Array xs;
        Array yc;
        Array ys;
        Array zc;
        Array zs;
        int nfp;
        int mpol;
        int ntor;
        bool stellsym;

        SurfaceXYZFourier(int _mpol, int _ntor, int _nfp, bool _stellsym, vector<double> _quadpoints_phi, vector<double> _quadpoints_theta)
            : Surface<Array>(_quadpoints_phi, _quadpoints_theta), mpol(_mpol), ntor(_ntor), nfp(_nfp), stellsym(_stellsym) {
                numquadpoints_phi = quadpoints_phi.size();
                numquadpoints_theta = quadpoints_theta.size();
                xc = xt::zeros<double>({mpol, 2*ntor+1});
                xs = xt::zeros<double>({mpol, 2*ntor+1});
                yc = xt::zeros<double>({mpol, 2*ntor+1});
                ys = xt::zeros<double>({mpol, 2*ntor+1});
                zc = xt::zeros<double>({mpol, 2*ntor+1});
                zs = xt::zeros<double>({mpol, 2*ntor+1});
            }



        int num_dofs() override {
            //if(stellsym)
            //    return 2*mpol*(2*ntor+1) - ntor - (ntor+1);
            //else
                return 6*mpol*(2*ntor+1) - 3*ntor - 3*(ntor+1);
        }

        void set_dofs_impl(const vector<double>& dofs) override {
            int shift = mpol*(2*ntor+1);
            int counter = 0;
            if(stellsym) {
                //for (int i = ntor; i < shift; ++i)
                //    rc.data()[i] = dofs[counter++];
                //for (int i = ntor+1; i < shift; ++i)
                //    zs.data()[i] = dofs[counter++];

            } else {
                for (int i = ntor; i < shift; ++i)
                    xc.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    xs.data()[i] = dofs[counter++];
                for (int i = ntor; i < shift; ++i)
                    yc.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    ys.data()[i] = dofs[counter++];
                for (int i = ntor; i < shift; ++i)
                    zc.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    zs.data()[i] = dofs[counter++];
            }
        }

        vector<double> get_dofs() override {
            auto res = vector<double>(num_dofs(), 0.);
            int shift = mpol*(2*ntor+1);
            int counter = 0;
            if(stellsym) {
                //for (int i = ntor; i < shift; ++i)
                //    res[counter++] = rc[i];
                //for (int i = ntor+1; i < shift; ++i)
                //    res[counter++] = zs[i];
            } else {
                for (int i = ntor; i < shift; ++i)
                    res[counter++] = xc[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = xs[i];
                for (int i = ntor; i < shift; ++i)
                    res[counter++] = yc[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = ys[i];
                for (int i = ntor; i < shift; ++i)
                    res[counter++] = zc[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = zs[i];
            }
            return res;
        }

        inline double get_coeff(int dim, bool cos, int m, int i) {
            if     (dim == 0 && cos)
                return xc(m, i);
            else if(dim == 0 && !cos)
                return xs(m, i);
            else if(dim == 1 && cos)
                return yc(m, i);
            else if(dim == 1 && !cos)
                return ys(m, i);
            else if(dim == 2 && cos)
                return zc(m, i);
            else
                return zs(m, i);
        }
        void gamma_impl(Array& data) override {
            data *= 0.;
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    for (int m = 0; m < mpol; ++m) {
                        for (int i = 0; i < 2*ntor+1; ++i) {
                            int n  = i - ntor;
                            for (int d = 0; d < 3; ++d) {
                                data(k1, k2, d) += get_coeff(d, true , m, i) * cos(m*theta-n*nfp*phi);
                                data(k1, k2, d) += get_coeff(d, false, m, i) * sin(m*theta-n*nfp*phi);
                            }
                        }
                    }
                }
            }
        }

        void gammadash1_impl(Array& data) override {
            data *= 0.;
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    for (int m = 0; m < mpol; ++m) {
                        for (int i = 0; i < 2*ntor+1; ++i) {
                            int n  = i - ntor;
                            for (int d = 0; d < 3; ++d) {
                                data(k1, k2, d) += get_coeff(d, true , m, i) * (2*M_PI*n*nfp) *sin(m*theta-n*nfp*phi);
                                data(k1, k2, d) += get_coeff(d, false, m, i) * (-2*M_PI*n*nfp)*cos(m*theta-n*nfp*phi);
                            }
                        }
                    }
                }
            }
        }
        void gammadash2_impl(Array& data) override {
            data *= 0.;
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    for (int m = 0; m < mpol; ++m) {
                        for (int i = 0; i < 2*ntor+1; ++i) {
                            int n  = i - ntor;
                            for (int d = 0; d < 3; ++d) {
                                data(k1, k2, d) += get_coeff(d, true , m, i) * (-2*M_PI*m)*sin(m*theta-n*nfp*phi);
                                data(k1, k2, d) += get_coeff(d, false, m, i) * (2*M_PI*m)* cos(m*theta-n*nfp*phi);
                            }
                        }
                    }
                }
            }
        }

        void dgamma_by_dcoeff_impl(Array& data) override {
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    int counter = 0;
                    for (int d = 0; d < 3; ++d) {
                        for (int m = 0; m < mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<0) continue;
                                data(k1, k2, d, counter++) = cos(m*theta-n*nfp*phi);
                            }
                        }
                        for (int m = 0; m < mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<=0) continue;
                                data(k1, k2, d, counter++) = sin(m*theta-n*nfp*phi);
                            }
                        }
                    }
                }
            }
        }

        void dgammadash1_by_dcoeff_impl(Array& data) override {
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    int counter = 0;
                    for (int d = 0; d < 3; ++d) {
                        for (int m = 0; m < mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<0) continue;
                                data(k1, k2, d, counter++) = (2*M_PI*n*nfp) *sin(m*theta-n*nfp*phi);
                            }
                        }
                        for (int m = 0; m < mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<=0) continue;
                                data(k1, k2, d, counter++) = (-2*M_PI*n*nfp)*cos(m*theta-n*nfp*phi);
                            }
                        }
                    }
                }
            }
        }

        void dgammadash2_by_dcoeff_impl(Array& data) override {
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    int counter = 0;
                    for (int d = 0; d < 3; ++d) {
                        for (int m = 0; m < mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<0) continue;
                                data(k1, k2, d, counter++) = (-2*M_PI*m)*sin(m*theta-n*nfp*phi);
                            }
                        }
                        for (int m = 0; m < mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<=0) continue;
                                data(k1, k2, d, counter++) = (2*M_PI*m)* cos(m*theta-n*nfp*phi);
                            }
                        }
                    }
                }
            }
        }

        //double surface_area() override {
        //    double area = 0.;
        //    auto n = this->normal();
        //    for (int i = 0; i < numquadpoints_phi; ++i) {
        //        for (int j = 0; j < numquadpoints_theta; ++j) {
        //            area += sqrt(n(i,j,0)*n(i,j,0) + n(i,j,1)*n(i,j,1) + n(i,j,2)*n(i,j,2));
        //        }
        //    }
        //    return area/(numquadpoints_phi*numquadpoints_theta);
        //}

        //void dsurface_area_by_dcoeff_impl(Array& data) override {
        //    data *= 0.;
        //    auto n = this->normal();
        //    auto dn_dc = this->dnormal_by_dcoeff();
        //    int ndofs = num_dofs();
        //    for (int i = 0; i < numquadpoints_phi; ++i) {
        //        for (int j = 0; j < numquadpoints_theta; ++j) {
        //            for (int m = 0; m < ndofs; ++m) {
        //                data(m) += 0.5 * (dn_dc(i,j,0,m)*n(i,j,0) + dn_dc(i,j,1,m)*n(i,j,1) + dn_dc(i,j,2,m)*n(i,j,2)) / sqrt(n(i,j,0)*n(i,j,0) + n(i,j,1)*n(i,j,1) + n(i,j,2)*n(i,j,2));
        //            }
        //        }
        //    }
        //    data *= 1./ (numquadpoints_phi*numquadpoints_theta);
        //}

};
