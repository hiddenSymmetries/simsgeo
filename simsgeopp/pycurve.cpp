#pragma once

#include "curve.cpp"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;

class PyCurve : public Curve<PyArray> {
    using Curve<PyArray>::Curve;

    public:
        virtual int num_dofs() override {
            PYBIND11_OVERLOAD_PURE(int, Curve, num_dofs);
        }

        virtual void set_dofs(const vector<double>& _dofs) override {
            PYBIND11_OVERLOAD_PURE(void, Curve, set_dofs, _dofs);
        }

        virtual vector<double> get_dofs() override {
            PYBIND11_OVERLOAD_PURE(vector<double>, Curve, get_dofs);
        }

        virtual void gamma_impl(PyArray& data) override {
            PYBIND11_OVERLOAD_PURE(void, Curve, gamma_impl, data);
        }

        virtual void dgamma_by_dphi_impl(PyArray& data) override {
            PYBIND11_OVERLOAD_PURE(void, Curve, dgamma_by_dphi_impl, data);
        }

        virtual void dgamma_by_dcoeff_impl(PyArray& data) override {
            PYBIND11_OVERLOAD_PURE(void, Curve, dgamma_by_dcoeff_impl, data);
        }
};
