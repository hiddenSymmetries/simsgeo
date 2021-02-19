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
#include "cachedarray.hpp"


template<class Array>
class Surface {
    private:
        map<string, CachedArray<Array>> cache;

        Array& check_the_cache(string key, vector<int> dims, std::function<void(Array&)> impl){
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>(dims)))).first; 
            }
            if(!((loc->second).status)){ // needs recomputing
                impl((loc->second).data);
                (loc->second).status = true;
            }
            return (loc->second).data;
        }

    // We'd really like these to be protected, but I'm not sure that plays well
    // with accessing them from python child classes. 
    public://protected:
        int numquadpoints_phi;
        int numquadpoints_theta;
        vector<double> quadpoints_phi;
        vector<double> quadpoints_theta;

    public:

        Surface(vector<double> _quadpoints_phi, vector<double> _quadpoints_theta) : quadpoints_phi(_quadpoints_phi),quadpoints_theta( _quadpoints_theta) {
            numquadpoints_phi = quadpoints_phi.size();
            numquadpoints_theta = quadpoints_theta.size();
        }

        void invalidate_cache() {

            for (auto it = cache.begin(); it != cache.end(); ++it) {
                (it->second).status = false;
            }
        }

        void set_dofs(const vector<double>& _dofs) {
            this->set_dofs_impl(_dofs);
            this->invalidate_cache();
        }

        virtual int num_dofs() = 0;
        virtual void set_dofs_impl(const vector<double>& _dofs) = 0;
        virtual vector<double> get_dofs() = 0;

        virtual void gamma_impl(Array& data) = 0;
        virtual void gammadash1_impl(Array& data) = 0;
        virtual void gammadash2_impl(Array& data) = 0;
        virtual void normal_impl(Array& data) = 0;
        virtual double surface_area() = 0;

        Array& gamma() {
            return check_the_cache("gamma", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return gamma_impl(A);});
        }
        Array& gammadash1() {
            return check_the_cache("gammadash1", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return gammadash1_impl(A);});
        }
        Array& gammadash2() {
            return check_the_cache("gammadash2", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return gammadash2_impl(A);});
        }
        Array& normal() {
            return check_the_cache("normal", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return normal_impl(A);});
        }



        virtual ~Surface() = default;
};
