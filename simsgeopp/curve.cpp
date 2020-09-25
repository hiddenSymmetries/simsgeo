#pragma once
#include <vector>
using std::vector;

#include <string>
using std::string;

#include <map> 
using std::map;
#include <stdexcept>
using std::logic_error;

#include "xtensor/xarray.hpp"

template<class Array>
struct CachedArray {
    Array data;
    bool status;
    CachedArray(Array _data) : data(_data), status(false) {}
};

template<class Array>
Array curve_vjp_contraction(const Array& mat, const Array& v){
    int numquadpoints = mat.shape(0);
    int numdofs = mat.shape(2);
    Array res = xt::zeros<double>({numdofs});
    for (int i = 0; i < numdofs; ++i) {
        for (int j = 0; j < numquadpoints; ++j) {
            for (int k = 0; k < 3; ++k) {
                res(i) += mat(j, k, i) * v(j, k);
            }
        }
    }
    return res;
}

template<class Array>
class Curve {
    private:
        map<string, CachedArray<Array>> cache;

        Array& check_the_cache(string key, vector<int> dims, std::function<void(Array&)> impl){
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                std::cout << "Allocate memory for '" + key + "'" << std::endl;
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>(dims)))).first; 
            }
            if(!((loc->second).status)){ // needs recomputing
                std::cout << "Call implementation for '" + key + "'" << std::endl;
                impl((loc->second).data);
                (loc->second).status = true;
            }
            return (loc->second).data;
        }

    // We'd really like these to be protected, but I'm not sure that plays well
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
        virtual void set_dofs(const vector<double>& _dofs) = 0;
        virtual vector<double> get_dofs() = 0;

        virtual void gamma_impl(Array& data) = 0;
        virtual void gammadash_impl(Array& data) { throw logic_error("gammadash_impl was not implemented"); };
        virtual void gammadashdash_impl(Array& data) { throw logic_error("gammadashdash_impl was not implemented"); };
        virtual void gammadashdashdash_impl(Array& data) { throw logic_error("gammadashdashdash_impl was not implemented"); };

        virtual void dgamma_by_dcoeff_impl(Array& data) { throw logic_error("dgamma_by_dcoeff_impl was not implemented"); };
        virtual void dgammadash_by_dcoeff_impl(Array& data) { throw logic_error("dgammadash_by_dcoeff_impl was not implemented"); };
        virtual void dgammadashdash_by_dcoeff_impl(Array& data) { throw logic_error("dgammadashdash_by_dcoeff_impl was not implemented"); };
        virtual void dgammadashdashdash_by_dcoeff_impl(Array& data) { throw logic_error("dgammadashdashdash_by_dcoeff_impl was not implemented"); };

        virtual void kappa_impl(Array& data) { throw logic_error("kappa_impl was not implemented"); };
        virtual void dkappa_by_dcoeff_impl(Array& data) { throw logic_error("dkappa_by_dcoeff_impl was not implemented"); };

        virtual void torsion_impl(Array& data) { throw logic_error("torsion_impl was not implemented"); };
        virtual void dtorsion_by_dcoeff_impl(Array& data) { throw logic_error("dtorsion_by_dcoeff_impl was not implemented"); };


        Array& gamma() {
            return check_the_cache("gamma", {numquadpoints, 3}, [this](Array& A) { return gamma_impl(A);});
        }
        Array& gammadash() {
            return check_the_cache("gammadash", {numquadpoints, 3}, [this](Array& A) { return gammadash_impl(A);});
        }
        Array& gammadashdash() {
            return check_the_cache("gammadashdash", {numquadpoints, 3}, [this](Array& A) { return gammadashdash_impl(A);});
        }
        Array& gammadashdashdash() {
            return check_the_cache("gammadashdashdash", {numquadpoints, 3}, [this](Array& A) { return gammadashdashdash_impl(A);});
        }

        Array& dgamma_by_dcoeff() {
            return check_the_cache("dgamma_by_dcoeff", {numquadpoints, 3, num_dofs()}, [this](Array& A) { return dgamma_by_dcoeff_impl(A);});
        }
        Array& dgammadash_by_dcoeff() {
            return check_the_cache("dgammadash_by_dcoeff", {numquadpoints, 3, num_dofs()}, [this](Array& A) { return dgammadash_by_dcoeff_impl(A);});
        }
        Array& dgammadashdash_by_dcoeff() {
            return check_the_cache("dgammadashdash_by_dcoeff", {numquadpoints, 3, num_dofs()}, [this](Array& A) { return dgammadashdash_by_dcoeff_impl(A);});
        }
        Array& dgammadashdashdash_by_dcoeff() {
            return check_the_cache("dgammadashdashdash_by_dcoeff", {numquadpoints, 3, num_dofs()}, [this](Array& A) { return dgammadashdashdash_by_dcoeff_impl(A);});
        }

        virtual Array dgamma_by_dcoeff_vjp(Array& v) {
            return curve_vjp_contraction<Array>(dgamma_by_dcoeff(), v);
        };

        virtual Array dgammadash_by_dcoeff_vjp(Array& v) {
            return curve_vjp_contraction<Array>(dgammadash_by_dcoeff(), v);
        };

        virtual Array dgammadashdash_by_dcoeff_vjp(Array& v) {
            return curve_vjp_contraction<Array>(dgammadashdash_by_dcoeff(), v);
        };

        virtual Array dgammadashdashdash_by_dcoeff_vjp(Array& v) {
            return curve_vjp_contraction<Array>(dgammadashdashdash_by_dcoeff(), v);
        };

        Array& kappa() {
            return check_the_cache("kappa", {numquadpoints}, [this](Array& A) { return kappa_impl(A);});
        }

        Array& torsion() {
            return check_the_cache("torsion", {numquadpoints}, [this](Array& A) { return torsion_impl(A);});
        }

        virtual ~Curve() = default;
};
