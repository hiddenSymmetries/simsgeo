#pragma once

#include "curve.cpp"

template<class Array>
class CurveRZFourier : public Curve<Array> {
    private:
        int order;
        int nfp;
        bool stellsym;
    public:
        using Curve<Array>::quadpoints;
        using Curve<Array>::numquadpoints;
        vector<vector<double>> dofs;

        Array rc;
        Array rs;
        Array zc;
        Array zs;

        CurveRZFourier(int _numquadpoints, int _order, int _nfp, bool _stellsym) : Curve<Array>(std::vector<double>(_numquadpoints, 0.)), order(_order), nfp(_nfp), stellsym(_stellsym) {
            for (int i = 0; i < numquadpoints; ++i) {
                this->quadpoints[i] = ((double)i)/(nfp*numquadpoints);
            }
            rc = xt::zeros<double>({order + 1});
            rs = xt::zeros<double>({order});
            zc = xt::zeros<double>({order + 1});
            zs = xt::zeros<double>({order});
        }

        CurveRZFourier(vector<double> _quadpoints, int _order, int _nfp, bool _stellsym) : Curve<Array>(_quadpoints), order(_order), nfp(_nfp), stellsym(_stellsym) {
            rc = xt::zeros<double>({order + 1});
            rs = xt::zeros<double>({order});
            zc = xt::zeros<double>({order + 1});
            zs = xt::zeros<double>({order});
        }

        inline int get_nfp() {
            return nfp;
        }

        inline int num_dofs() override {
            if(stellsym)
                return 2*order+1;
            else
                return 2*(2*order+1);
        }

        void set_dofs_impl(const vector<double>& dofs) override {
            int counter = 0;
            if(stellsym) {
                for (int i = 0; i < order + 1; ++i)
                    rc.data()[i] = dofs[counter++];
                for (int i = 0; i < order; ++i)
                    zs.data()[i] = dofs[counter++];
            } else {
                for (int i = 0; i < order + 1; ++i)
                    rc.data()[i] = dofs[counter++];
                for (int i = 0; i < order; ++i)
                    rs.data()[i] = dofs[counter++];
                for (int i = 0; i < order + 1; ++i)
                    zc.data()[i] = dofs[counter++];
                for (int i = 0; i < order; ++i)
                    zs.data()[i] = dofs[counter++];
            }
        }

        vector<double> get_dofs() override {
            auto res = vector<double>(num_dofs(), 0.);
            int counter = 0;
            if(stellsym) {
                for (int i = 0; i < order + 1; ++i)
                    res[counter++] = rc[i];
                for (int i = 0; i < order; ++i)
                    res[counter++] = zs[i];
            } else {
                for (int i = 0; i < order + 1; ++i)
                    res[counter++] = rc[i];
                for (int i = 0; i < order; ++i)
                    res[counter++] = rs[i];
                for (int i = 0; i < order + 1; ++i)
                    res[counter++] = zc[i];
                for (int i = 0; i < order; ++i)
                    res[counter++] = zs[i];
            }
            return res;
        }

        void gamma_impl(Array& data) override {
            data *= 0;
            for (int k = 0; k < numquadpoints; ++k) {
                double phi = 2 * M_PI * quadpoints[k];
                for (int i = 0; i < order+1; ++i) {
                    data(k, 0) += rc[i] * cos(nfp*i*phi) * cos(phi);
                    data(k, 1) += rc[i] * cos(nfp*i*phi) * sin(phi);
                }
                for (int i = 1; i < order+1; ++i) {
                    data(k, 2) += zs[i-1] * sin(nfp*i*phi);
                }
            }
            if(!stellsym){
                for (int k = 0; k < numquadpoints; ++k) {
                    double phi = 2 * M_PI * quadpoints[k];
                    for (int i = 1; i < order+1; ++i) {
                        data(k, 0) += rs[i-1] * sin(nfp*i*phi) * cos(phi);
                        data(k, 1) += rs[i-1] * sin(nfp*i*phi) * sin(phi);
                    }
                    for (int i = 0; i < order+1; ++i) {
                        data(k, 2) += zc[i] * cos(nfp*i*phi);
                    }
                }
            }
        }

        void gammadash_impl(Array& data) override {
            data *= 0;
            for (int k = 0; k < numquadpoints; ++k) {
                double phi = 2 * M_PI * quadpoints[k];
                for (int i = 0; i < order+1; ++i) {
                    data(k, 0) += rc[i] * ( -(i*nfp) * sin(nfp*i*phi) * cos(phi) - cos(nfp*i*phi) * sin(phi));
                    data(k, 1) += rc[i] * ( -(i*nfp) * sin(nfp*i*phi) * sin(phi) + cos(nfp*i*phi) * cos(phi));
                }
                for (int i = 1; i < order+1; ++i) {
                    data(k, 2) += zs[i-1] * (nfp*i) * cos(nfp*i*phi);
                }
            }
            if(!stellsym){
                for (int k = 0; k < numquadpoints; ++k) {
                    double phi = 2 * M_PI * quadpoints[k];
                    for (int i = 1; i < order+1; ++i) {
                        data(k, 0) += rs[i-1] * ( (i*nfp) * cos(nfp*i*phi) * cos(phi) - sin(nfp*i*phi) * sin(phi));
                        data(k, 1) += rs[i-1] * ( (i*nfp) * cos(nfp*i*phi) * sin(phi) + sin(nfp*i*phi) * cos(phi));
                    }
                    for (int i = 0; i < order+1; ++i) {
                        data(k, 2) -= zc[i] * (nfp*i) * sin(nfp*i*phi);
                    }
                }
            }
            data *= (2*M_PI);
        }

        void gammadashdash_impl(Array& data) override {
            data *= 0;
            for (int k = 0; k < numquadpoints; ++k) {
                double phi = 2 * M_PI * quadpoints[k];
                for (int i = 0; i < order+1; ++i) {
                    data(k, 0) += rc[i] * (+2*(nfp*i)*sin(nfp*i*phi)*sin(phi)-(pow(nfp*i, 2)+1)*cos(nfp*i*phi)*cos(phi));
                    data(k, 1) += rc[i] * (-2*(nfp*i)*sin(nfp*i*phi)*cos(phi)-(pow(nfp*i, 2)+1)*cos(nfp*i*phi)*sin(phi));
                }
                for (int i = 1; i < order+1; ++i) {
                    data(k, 2) -= zs[i-1] * pow(nfp*i, 2)*sin(nfp*i*phi);
                }
            }
            if(!stellsym){
                for (int k = 0; k < numquadpoints; ++k) {
                    double phi = 2 * M_PI * quadpoints[k];
                    for (int i = 1; i < order+1; ++i) {
                        data(k, 0) += rs[i-1] * (-(pow(nfp*i,2)+1)*sin(nfp*i*phi)*cos(phi) - 2*(i*nfp)*cos(nfp*i*phi)*sin(phi));
                        data(k, 1) += rs[i-1] * (-(pow(nfp*i,2)+1)*sin(nfp*i*phi)*sin(phi) + 2*(i*nfp)*cos(nfp*i*phi)*cos(phi));
                    }
                    for (int i = 0; i < order+1; ++i) {
                        data(k, 2) -= zc[i] * pow(nfp*i, 2)*cos(nfp*i*phi);
                    }
                }
            }
            data *= 2*M_PI*2*M_PI;
        }

        void gammadashdashdash_impl(Array& data) override {
            data *= 0;
            for (int k = 0; k < numquadpoints; ++k) {
                double phi = 2 * M_PI * quadpoints[k];
                for (int i = 0; i < order+1; ++i) {
                    data(k, 0) += rc[i]*(
                            +2*pow(nfp*i, 2)*cos(nfp*i*phi)*sin(phi)
                            +2*(nfp*i)*sin(nfp*i*phi)*cos(phi)
                            +(pow(nfp*i, 2) + 1)*(nfp*i)*sin(nfp*i*phi)*cos(phi)
                            +(pow(nfp*i, 2) + 1)*cos(nfp*i*phi)*sin(phi)
                            );
                    data(k, 1) += rc[i]*(
                            -2*pow(nfp*i, 2)*cos(nfp*i*phi)*cos(phi)
                            +2*(nfp*i)*   sin(nfp*i*phi)*sin(phi)
                            +(pow(nfp*i, 2) + 1)*(nfp*i)*sin(nfp*i*phi)*sin(phi)
                            -(pow(nfp*i, 2) + 1)*cos(nfp*i*phi)*cos(phi)
                            );
                }
                for (int i = 1; i < order+1; ++i) {
                    data(k, 2) -= zs[i-1] * pow(nfp*i, 3) * cos(nfp*i*phi);
                }
            }
            if(!stellsym){
                for (int k = 0; k < numquadpoints; ++k) {
                    double phi = 2 * M_PI * quadpoints[k];
                    for (int i = 1; i < order+1; ++i) {
                        data(k, 0) += rs[i-1]*(
                                -(pow(nfp*i,2)+1) * (nfp*i) * cos(nfp*i*phi)*cos(phi)
                                +(pow(nfp*i,2)+1) * sin(nfp*i*phi)*sin(phi)
                                +2*pow(nfp*i, 2)*sin(nfp*i*phi)*sin(phi)
                                -2*(i*nfp)*cos(nfp*i*phi)*cos(phi)
                                );
                        data(k, 1) += rs[i-1]*(
                                -(pow(nfp*i,2)+1)*(nfp*i)*cos(nfp*i*phi)*sin(phi) 
                                -(pow(nfp*i,2)+1)*sin(nfp*i*phi)*cos(phi) 
                                - 2*pow(nfp*i,2)*sin(nfp*i*phi)*cos(phi)
                                - 2*(nfp*i)*cos(nfp*i*phi)*sin(phi)
                                );
                    }
                    for (int i = 0; i < order+1; ++i) {
                        data(k, 2) += zc[i] * pow(nfp*i, 3) * sin(nfp*i*phi);
                    }
                }
            }
            data *= 2*M_PI*2*M_PI*2*M_PI;
        }

        void dgamma_by_dcoeff_impl(Array& data) override {
            for (int k = 0; k < numquadpoints; ++k) {
                double phi = 2 * M_PI * quadpoints[k];
                int counter = 0;
                for (int i = 0; i < order+1; ++i) {
                    data(k, 0, counter) = cos(nfp*i*phi) * cos(phi);
                    data(k, 1, counter) = cos(nfp*i*phi) * sin(phi);
                    counter++;
                }
                if(!stellsym){
                    for (int i = 1; i < order+1; ++i) {
                        data(k, 0, counter) = sin(nfp*i*phi) * cos(phi);
                        data(k, 1, counter) = sin(nfp*i*phi) * sin(phi);
                        counter++;
                    }
                    for (int i = 0; i < order+1; ++i) {
                        data(k, 2, counter) = cos(nfp*i*phi);
                        counter++;
                    }
                }
                for (int i = 1; i < order+1; ++i) {
                    data(k, 2, counter) = sin(nfp*i*phi);
                    counter++;
                }
            }
        }

        void dgammadash_by_dcoeff_impl(Array& data) override {
            for (int k = 0; k < numquadpoints; ++k) {
                double phi = 2 * M_PI * quadpoints[k];
                int counter = 0;
                for (int i = 0; i < order+1; ++i) {
                    data(k, 0, counter) = ( -(i*nfp) * sin(nfp*i*phi) * cos(phi) - cos(nfp*i*phi) * sin(phi));
                    data(k, 1, counter) = ( -(i*nfp) * sin(nfp*i*phi) * sin(phi) + cos(nfp*i*phi) * cos(phi));
                    counter++;
                }
                if(!stellsym){
                    for (int i = 1; i < order+1; ++i) {
                        data(k, 0, counter) = ( (i*nfp) * cos(nfp*i*phi) * cos(phi) - sin(nfp*i*phi) * sin(phi));
                        data(k, 1, counter) = ( (i*nfp) * cos(nfp*i*phi) * sin(phi) + sin(nfp*i*phi) * cos(phi));
                        counter++;
                    }
                    for (int i = 0; i < order+1; ++i) {
                        data(k, 2, counter) = -(nfp*i) * sin(nfp*i*phi);
                        counter++;
                    }
                }
                for (int i = 1; i < order+1; ++i) {
                    data(k, 2, counter) = (nfp*i) * cos(nfp*i*phi);
                    counter++;
                }
            }
            data *= (2*M_PI);
        }

        void dgammadashdash_by_dcoeff_impl(Array& data) override {
            for (int k = 0; k < numquadpoints; ++k) {
                double phi = 2 * M_PI * quadpoints[k];
                int counter = 0;
                for (int i = 0; i < order+1; ++i) {
                    data(k, 0, counter) = (+2*(nfp*i)*sin(nfp*i*phi)*sin(phi)-(pow(nfp*i, 2)+1)*cos(nfp*i*phi)*cos(phi));
                    data(k, 1, counter) = (-2*(nfp*i)*sin(nfp*i*phi)*cos(phi)-(pow(nfp*i, 2)+1)*cos(nfp*i*phi)*sin(phi));
                    counter++;
                }
                if(!stellsym){
                    for (int i = 1; i < order+1; ++i) {
                        data(k, 0, counter) = (-(pow(nfp*i,2)+1)*sin(nfp*i*phi)*cos(phi) - 2*(i*nfp)*cos(nfp*i*phi)*sin(phi));
                        data(k, 1, counter) = (-(pow(nfp*i,2)+1)*sin(nfp*i*phi)*sin(phi) + 2*(i*nfp)*cos(nfp*i*phi)*cos(phi));
                        counter++;
                    }
                    for (int i = 0; i < order+1; ++i) {
                        data(k, 2, counter) = -pow(nfp*i, 2)*cos(nfp*i*phi);
                        counter++;
                    }
                }
                for (int i = 1; i < order+1; ++i) {
                    data(k, 2, counter) = -pow(nfp*i, 2)*sin(nfp*i*phi);
                    counter++;
                }
            }
            data *= 2*M_PI*2*M_PI;
        }

        void dgammadashdashdash_by_dcoeff_impl(Array& data) override {
            for (int k = 0; k < numquadpoints; ++k) {
                double phi = 2 * M_PI * quadpoints[k];
                int counter = 0;
                for (int i = 0; i < order+1; ++i) {
                    data(k, 0, counter) = (
                            +2*pow(nfp*i, 2)*cos(nfp*i*phi)*sin(phi)
                            +2*(nfp*i)*sin(nfp*i*phi)*cos(phi)
                            +(pow(nfp*i, 2) + 1)*(nfp*i)*sin(nfp*i*phi)*cos(phi)
                            +(pow(nfp*i, 2) + 1)*cos(nfp*i*phi)*sin(phi)
                            );
                    data(k, 1, counter) = (
                            -2*pow(nfp*i, 2)*cos(nfp*i*phi)*cos(phi)
                            +2*(nfp*i)*   sin(nfp*i*phi)*sin(phi)
                            +(pow(nfp*i, 2) + 1)*(nfp*i)*sin(nfp*i*phi)*sin(phi)
                            -(pow(nfp*i, 2) + 1)*cos(nfp*i*phi)*cos(phi)
                            );
                    counter++;
                }
                if(!stellsym){
                    for (int i = 1; i < order+1; ++i) {
                        data(k, 0, counter) = (
                                -(pow(nfp*i,2)+1) * (nfp*i) * cos(nfp*i*phi)*cos(phi)
                                +(pow(nfp*i,2)+1) * sin(nfp*i*phi)*sin(phi)
                                +2*pow(nfp*i, 2)*sin(nfp*i*phi)*sin(phi)
                                -2*(i*nfp)*cos(nfp*i*phi)*cos(phi)
                                );
                        data(k, 1, counter) = (
                                -(pow(nfp*i,2)+1)*(nfp*i)*cos(nfp*i*phi)*sin(phi) 
                                -(pow(nfp*i,2)+1)*sin(nfp*i*phi)*cos(phi) 
                                - 2*pow(nfp*i,2)*sin(nfp*i*phi)*cos(phi)
                                - 2*(nfp*i)*cos(nfp*i*phi)*sin(phi)
                                );
                        counter++;
                    }
                    for (int i = 0; i < order+1; ++i) {
                        data(k, 2, counter) = pow(nfp*i, 3) * sin(nfp*i*phi);
                        counter++;
                    }
                }

                for (int i = 1; i < order+1; ++i) {
                    data(k, 2, counter) = -pow(nfp*i, 3) * cos(nfp*i*phi);
                    counter++;
                }
            }
            data *= 2*M_PI*2*M_PI*2*M_PI;
        }

};
