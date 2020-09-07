#pragma once
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
        virtual void dgamma_by_dphi_impl(Array& data) = 0;
        virtual void dgamma_by_dcoeff_impl(Array& data) = 0;

        Array& gamma() {
            return check_the_cache("gamma", {numquadpoints, 3}, [this](Array& A) { return gamma_impl(A);});
        }

        Array& dgamma_by_dphi() {
            return check_the_cache("dgamma_by_dphi", {1, numquadpoints, 3}, [this](Array& A) { return dgamma_by_dphi_impl(A);});
        }

        Array& dgamma_by_dcoeff() {
            return check_the_cache("dgamma_by_dcoeff", {num_dofs(), numquadpoints, 3}, [this](Array& A) { return dgamma_by_dcoeff_impl(A);});
        }

        virtual ~Curve() = default;
};
