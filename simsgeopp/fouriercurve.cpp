#pragma once

#include "curve.cpp"

template<class Array>
class FourierCurve : public Curve<Array> {
    private:
        vector<vector<double>> dofs;
        int order;
        using Curve<Array>::numquadpoints;
        using Curve<Array>::quadpoints;
    public:
        FourierCurve(int _numquadpoints, int _order) : Curve<Array>(_numquadpoints), order(_order) {
            dofs = vector<vector<double>> {
                vector<double>(2*order+1, 0.), 
                vector<double>(2*order+1, 0.), 
                vector<double>(2*order+1, 0.)
            };
        }

        inline int num_dofs() {
            return 3*(2*order+1);
        }

        void set_dofs(const vector<double>& _dofs) {
            int counter = 0;
            for (int i = 0; i < 3; ++i) {
                dofs[i][0] = _dofs[counter++];
                for (int j = 1; j < order+1; ++j) {
                    dofs[i][2*j-1] = _dofs[counter++];
                    dofs[i][2*j] = _dofs[counter++];
                }
            }
        }

        vector<double> get_dofs() {
            auto _dofs = vector<double>(num_dofs(), 0.);
            int counter = 0;
            for (int i = 0; i < 3; ++i) {
                _dofs[counter++] = dofs[i][0];
                for (int j = 1; j < order+1; ++j) {
                    _dofs[counter++] = dofs[i][2*j-1];
                    _dofs[counter++] = dofs[i][2*j];
                }
            }
            return _dofs;
        }

        void gamma_impl(Array& data) {
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < 3; ++i) {
                    data(k, i) += dofs[i][0];
                    for (int j = 1; j < order+1; ++j) {
                        data(k, i) += dofs[i][2*j-1]*sin(2*M_PI*j*quadpoints[k]);
                        data(k, i) += dofs[i][2*j]*cos(2*M_PI*j*quadpoints[k]);
                    }
                }
            }
        }

        void dgamma_by_dphi_impl(Array& data) {
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 1; j < order+1; ++j) {
                        data(0, k, i) += +dofs[i][2*j-1]*2*M_PI*j*cos(2*M_PI*j*quadpoints[k]);
                        data(0, k, i) += -dofs[i][2*j]*2*M_PI*j*sin(2*M_PI*j*quadpoints[k]);
                    }
                }
            }
        }

        void dgamma_by_dcoeff_impl(Array& data) {
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < 3; ++i) {
                    data(k, i, i*(2*order+1)) = 1.;
                    for (int j = 1; j < order+1; ++j) {
                        data(i*(2*order+1) + 2*j-1, k, i) = sin(2*M_PI*j*quadpoints[k]);
                        data(i*(2*order+1) + 2*j  , k, i) = cos(2*M_PI*j*quadpoints[k]);
                    }
                }
            }
        }
};

