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

        virtual void gammadash_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, Curve, gammadash_impl, data);
        }

        virtual void gammadashdash_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, Curve, gammadashdash_impl, data);
        }

        virtual void gammadashdashdash_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, Curve, gammadashdashdash_impl, data);
        }

        virtual void dgamma_by_dcoeff_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, Curve, dgamma_by_dcoeff_impl, data);
        }

        virtual void dgammadash_by_dcoeff_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, Curve, dgammadash_by_dcoeff_impl, data);
        }

        virtual void dgammadashdash_by_dcoeff_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, Curve, dgammadashdash_by_dcoeff_impl, data);
        }

        virtual void dgammadashdashdash_by_dcoeff_impl(PyArray& data) override {
            PYBIND11_OVERLOAD(void, Curve, dgammadashdashdash_by_dcoeff_impl, data);
        }

        virtual PyArray dgamma_by_dcoeff_vjp(PyArray& v) override {
            PYBIND11_OVERLOAD(PyArray, Curve, dgamma_by_dcoeff_vjp, v);
        }

        virtual PyArray dgammadash_by_dcoeff_vjp(PyArray& v) override {
            PYBIND11_OVERLOAD(PyArray, Curve, dgammadash_by_dcoeff_vjp, v);
        }

        virtual PyArray dgammadashdash_by_dcoeff_vjp(PyArray& v) override {
            PYBIND11_OVERLOAD(PyArray, Curve, dgammadashdash_by_dcoeff_vjp, v);
        }

        virtual PyArray dgammadashdashdash_by_dcoeff_vjp(PyArray& v) override {
            PYBIND11_OVERLOAD(PyArray, Curve, dgammadashdashdash_by_dcoeff_vjp, v);
        }

        //virtual void kappa_impl(PyArray& data) override {
        //    PYBIND11_OVERLOAD(void, Curve, kappa_impl, data);
        //}

        //virtual void torsion_impl(PyArray& data) override {
        //    PYBIND11_OVERLOAD(void, Curve, torsion_impl, data);
        //}
};
