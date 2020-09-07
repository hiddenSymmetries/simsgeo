#include <vector>
using std::vector;

#include <string>
using std::string;

#include <map> 
using std::map;

#include "xtensor/xarray.hpp"

template<class Array>
struct CachedArray {
    Array data;
    bool status;
    CachedArray(Array _data) : data(_data), status(false) {}
};

template<class Array>
class Curve {
    private:
        map<string, CachedArray<Array>> cache;

    // We'd really like these to be protected, but I'm not sure that play well
    // with accessing them from python child classes. 
    public://protected:
        int numquadpoints;
        vector<double> quadpoints;

    public:

        Curve(int _numquadpoints) : numquadpoints(_numquadpoints) {
            quadpoints = std::vector(numquadpoints, 0.);
            for (int i = 0; i < numquadpoints; ++i) {
                quadpoints[i] = ((double)i)/numquadpoints;
            }
        }

        void invalidate_cache() {
            for (auto it = cache.begin(); it != cache.end(); ++it) {
                (it->second).status = false;
            }
        }

        virtual int num_dofs() = 0;
        virtual void set_dofs(vector<double>& _dofs) = 0;
        virtual vector<double> get_dofs() = 0;

        virtual void gamma_impl(Array& data) = 0;
        virtual void dgamma_by_dphi_impl(Array& data) = 0;
        virtual void dgamma_by_dcoeff_impl(Array& data) = 0;

        Array& gamma() {
            string key = "gamma";
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                std::cout << "Allocate memory for 'gamma'" << std::endl;
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>({numquadpoints, 3})))).first; 
            }
            if(!((loc->second).status)){ // needs recomputing
                gamma_impl((loc->second).data);
                (loc->second).status = true;
            }
            return (loc->second).data;
        }

        Array& dgamma_by_dphi() {
            string key = "dgamma_by_dphi";
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                std::cout << "Allocate memory for 'dgamma_by_dphi'" << std::endl;
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>({1, numquadpoints, 3})))).first; 
            }
            if(!((loc->second).status)){ // needs recomputing
                dgamma_by_dphi_impl((loc->second).data);
                (loc->second).status = true;
            }
            return (loc->second).data;
        }

        Array& dgamma_by_dcoeff() {
            string key = "dgamma_by_dcoeff";
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                std::cout << "Allocate memory for 'dgamma_by_dcoeff'" << std::endl;
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>({num_dofs(), numquadpoints, 3})))).first; 
            }
            if(!((loc->second).status)){ // needs recomputing
                dgamma_by_dcoeff_impl((loc->second).data);
                (loc->second).status = true;
            }
            return (loc->second).data;
        }

        virtual ~Curve() = default;
};

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

        void set_dofs(vector<double>& _dofs) {
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
            std::cout << "Call gamma_impl" << std::endl;
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
            std::cout << "Call dgamma_by_dphi_impl" << std::endl;
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
            std::cout << "Call dgamma_by_dcoeff_impl" << std::endl;
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
