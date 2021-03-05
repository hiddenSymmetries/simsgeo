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
        bool stellsym;

        SurfaceRZFourier(int _mpol, int _ntor, int _nfp, bool _stellsym, vector<double> _quadpoints_phi, vector<double> _quadpoints_theta)
            : Surface<Array>(_quadpoints_phi, _quadpoints_theta), mpol(_mpol), ntor(_ntor), nfp(_nfp), stellsym(_stellsym) {
                numquadpoints_phi = quadpoints_phi.size();
                numquadpoints_theta = quadpoints_theta.size();
                rc = xt::zeros<double>({mpol, 2*ntor+1});
                rs = xt::zeros<double>({mpol, 2*ntor+1});
                zc = xt::zeros<double>({mpol, 2*ntor+1});
                zs = xt::zeros<double>({mpol, 2*ntor+1});
            }

        int num_dofs() override {
            if(stellsym)
                return 2*mpol*(2*ntor+1);
            else
                return 4*mpol*(2*ntor+1);
        }

        void set_dofs_impl(const vector<double>& dofs) override {
            int shift = mpol*(2*ntor+1);
            if(stellsym) {
                for (int i = 0; i < shift; ++i)
                    rc.data()[i] = dofs[i];
                for (int i = 0; i < shift; ++i)
                    zs.data()[i] = dofs[shift + i];

            } else {
                for (int i = 0; i < shift; ++i)
                    rc.data()[i] = dofs[i];
                for (int i = 0; i < shift; ++i)
                    rs.data()[i] = dofs[shift + i];
                for (int i = 0; i < shift; ++i)
                    zc.data()[i] = dofs[2*shift + i];
                for (int i = 0; i < shift; ++i)
                    zs.data()[i] = dofs[3*shift + i];
            }
        }
        vector<double> get_dofs() override {
            auto res = vector<double>(num_dofs(), 0.);
            int shift = mpol*(2*ntor+1);
            if(stellsym) {
                for (int i = 0; i < shift; ++i)
                    res[i] = rc[i];
                for (int i = 0; i < shift; ++i)
                    res[shift + i] = zs[i];
            } else {
                for (int i = 0; i < shift; ++i)
                    res[i] = rc[i];
                for (int i = 0; i < shift; ++i)
                    res[shift + i] = rs[i];
                for (int i = 0; i < shift; ++i)
                    res[2*shift + i] = zc[i];
                for (int i = 0; i < shift; ++i)
                    res[3*shift + i] = zs[i];
            }
            return res;
        }

        void gamma_impl(Array& data) override {
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    double r = 0;
                    double z = 0;
                    for (int i = 0; i < 2*ntor+1; ++i) {
                        int n  = i - ntor;
                        for (int m = 0; m < mpol; ++m) {
                            r += rc(m, i) * cos(m*theta-n*nfp*phi);
                            if(!stellsym) {
                                r += rs(m, i) * sin(m*theta-n*nfp*phi);
                                z += zc(m, i) * cos(m*theta-n*nfp*phi);
                            }
                            z += zs(m, i) * sin(m*theta-n*nfp*phi);
                        }
                    }
                    data(k1, k2, 0) = r * cos(phi);
                    data(k1, k2, 1) = r * sin(phi);
                    data(k1, k2, 2) = z;
                }
            }
        }

        void gammadash1_impl(Array& data) override {
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    double r = 0;
                    double rd = 0;
                    double zd = 0;
                    for (int i = 0; i < 2*ntor+1; ++i) {
                        int n  = i - ntor;
                        for (int m = 0; m < mpol; ++m) {
                            r  += rc(m, i) * cos(m*theta-n*nfp*phi);
                            rd += rc(m, i) * (n*nfp) * sin(m*theta-n*nfp*phi);
                            if(!stellsym) {
                                r  += rs(m, i) * sin(m*theta-n*nfp*phi);
                                rd += rs(m, i) * (-n*nfp)*cos(m*theta-n*nfp*phi);
                                zd += zc(m, i) * (n*nfp)*sin(m*theta-n*nfp*phi);
                            }
                            zd += zs(m, i) * (-n*nfp)*cos(m*theta-n*nfp*phi);
                        }
                    }
                    data(k1, k2, 0) = 2*M_PI*(rd * cos(phi) - r * sin(phi));
                    data(k1, k2, 1) = 2*M_PI*(rd * sin(phi) + r * cos(phi));
                    data(k1, k2, 2) = 2*M_PI*zd;
                }
            }
        }
        void gammadash2_impl(Array& data) override {
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    double rd = 0;
                    double zd = 0;
                    for (int i = 0; i < 2*ntor+1; ++i) {
                        int n  = i - ntor;
                        for (int m = 0; m < mpol; ++m) {
                            rd += rc(m, i) * (-m) * sin(m*theta-n*nfp*phi);
                            if(!stellsym) {
                                rd += rs(m, i) * m * cos(m*theta-n*nfp*phi);
                                zd += zc(m, i) * (-m) * sin(m*theta-n*nfp*phi);
                            }
                            zd += zs(m, i) * m * cos(m*theta-n*nfp*phi);
                        }
                    }
                    data(k1, k2, 0) = 2*M_PI*rd*cos(phi);
                    data(k1, k2, 1) = 2*M_PI*rd*sin(phi);
                    data(k1, k2, 2) = 2*M_PI*zd;
                }
            }
        }

        void dgamma_by_dcoeff_impl(Array& data) override {
            int shift = (2*ntor+1)*mpol;
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    for (int i = 0; i < 2*ntor+1; ++i) {
                        int n  = i - ntor;
                        for (int m = 0; m < mpol; ++m) {
                            int dofidx = m*(2*ntor+1) + i;
                            int offset = 0;
                            data(k1, k2, 0, dofidx + offset) = cos(m*theta-n*nfp*phi) * cos(phi);
                            data(k1, k2, 1, dofidx + offset) = cos(m*theta-n*nfp*phi) * sin(phi);
                            data(k1, k2, 2, dofidx + offset) = 0;
                            offset += shift;
                            if(!stellsym) {
                                data(k1, k2, 0, dofidx + offset) = sin(m*theta-n*nfp*phi) * cos(phi);
                                data(k1, k2, 1, dofidx + offset) = sin(m*theta-n*nfp*phi) * sin(phi);
                                data(k1, k2, 2, dofidx + offset) = 0;
                                offset += shift;
                                data(k1, k2, 0, dofidx + offset) = 0;
                                data(k1, k2, 1, dofidx + offset) = 0;
                                data(k1, k2, 2, dofidx + offset) = cos(m*theta-n*nfp*phi);
                                offset += shift;
                            }
                            data(k1, k2, 0, dofidx + offset) = 0;
                            data(k1, k2, 1, dofidx + offset) = 0;
                            data(k1, k2, 2, dofidx + offset) = sin(m*theta-n*nfp*phi);
                        }
                    }
                }
            }
        }

        void dgammadash1_by_dcoeff_impl(Array& data) override {
            int shift = (2*ntor+1)*mpol;
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    for (int i = 0; i < 2*ntor+1; ++i) {
                        int n  = i - ntor;
                        for (int m = 0; m < mpol; ++m) {
                            int dofidx = m*(2*ntor+1) + i;
                            int offset = 0;

                            data(k1, k2, 0, dofidx + offset) = 2*M_PI*((n*nfp) * sin(m*theta-n*nfp*phi) * cos(phi) - cos(m*theta-n*nfp*phi) * sin(phi));
                            data(k1, k2, 1, dofidx + offset) = 2*M_PI*((n*nfp) * sin(m*theta-n*nfp*phi) * sin(phi) + cos(m*theta-n*nfp*phi) * cos(phi));
                            data(k1, k2, 2, dofidx + offset) = 0.;
                            offset += shift;
                            if(!stellsym) {
                                data(k1, k2, 0, dofidx + offset) = 2*M_PI*((-n*nfp)*cos(m*theta-n*nfp*phi) * cos(phi) - sin(m*theta-n*nfp*phi) * sin(phi));
                                data(k1, k2, 1, dofidx + offset) = 2*M_PI*((-n*nfp)*cos(m*theta-n*nfp*phi) * sin(phi) + sin(m*theta-n*nfp*phi) * cos(phi));
                                data(k1, k2, 2, dofidx + offset) = 0.;
                                offset += shift;
                                data(k1, k2, 0, dofidx + offset) = 0.;
                                data(k1, k2, 1, dofidx + offset) = 0.;
                                data(k1, k2, 2, dofidx + offset) = 2*M_PI*(n*nfp)*sin(m*theta-n*nfp*phi);
                                offset += shift;
                            }
                            data(k1, k2, 0, dofidx + offset) = 0.;
                            data(k1, k2, 1, dofidx + offset) = 0.;
                            data(k1, k2, 2, dofidx + offset) = 2*M_PI*(-n*nfp)*cos(m*theta-n*nfp*phi);
                        }
                    }
                }
            }
        }
        void dgammadash2_by_dcoeff_impl(Array& data) override {
            int shift = (2*ntor+1)*mpol;
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    for (int i = 0; i < 2*ntor+1; ++i) {
                        int n  = i - ntor;
                        for (int m = 0; m < mpol; ++m) {
                            int dofidx = m*(2*ntor+1) + i;
                            int offset = 0;

                            data(k1, k2, 0, dofidx + offset) = 2*M_PI*(-m) * sin(m*theta-n*nfp*phi)*cos(phi);
                            data(k1, k2, 1, dofidx + offset) = 2*M_PI*(-m) * sin(m*theta-n*nfp*phi)*sin(phi);
                            data(k1, k2, 2, dofidx + offset) = 0.;
                            offset += shift;

                            if(!stellsym) {
                                data(k1, k2, 0, dofidx + offset) = 2*M_PI*m * cos(m*theta-n*nfp*phi)*cos(phi);
                                data(k1, k2, 1, dofidx + offset) = 2*M_PI*m * cos(m*theta-n*nfp*phi)*sin(phi);
                                data(k1, k2, 2, dofidx + offset) = 0.;
                                offset += shift;

                                data(k1, k2, 0, dofidx + offset) = 0.;
                                data(k1, k2, 1, dofidx + offset) = 0.;
                                data(k1, k2, 2, dofidx + offset) = 2*M_PI*(-m) * sin(m*theta-n*nfp*phi);
                                offset += shift;
                            }

                            data(k1, k2, 0, dofidx + offset) = 0.;
                            data(k1, k2, 1, dofidx + offset) = 0.;
                            data(k1, k2, 2, dofidx + offset) = 2*M_PI*m * cos(m*theta-n*nfp*phi);
                        }
                    }
                }
            }
        }

        double surface_area() override {
            double area = 0.;
            auto n = this->normal();
            for (int i = 0; i < numquadpoints_phi; ++i) {
                for (int j = 0; j < numquadpoints_theta; ++j) {
                    area += sqrt(n(i,j,0)*n(i,j,0) + n(i,j,1)*n(i,j,1) + n(i,j,2)*n(i,j,2));
                }
            }
            return area/(numquadpoints_phi*numquadpoints_theta);
        }

        void dsurface_area_by_dcoeff_impl(Array& data) override {
            data *= 0.;
            auto n = this->normal();
            auto dn_dc = this->dnormal_by_dcoeff();
            int ndofs = num_dofs();
            for (int i = 0; i < numquadpoints_phi; ++i) {
                for (int j = 0; j < numquadpoints_theta; ++j) {
                    for (int m = 0; m < ndofs; ++m) {
                        data(m) += 0.5 * (dn_dc(i,j,0,m)*n(i,j,0) + dn_dc(i,j,1,m)*n(i,j,1) + dn_dc(i,j,2,m)*n(i,j,2)) \ sqrt(n(i,j,0)*n(i,j,0) + n(i,j,1)*n(i,j,1) + n(i,j,2)*n(i,j,2));
                    }
                }
            }
            data *= 1./ (numquadpoints_phi*numquadpoints_theta);
        }

};
