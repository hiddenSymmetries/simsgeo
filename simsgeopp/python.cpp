#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;

#include "surface.cpp"
#include "pysurface.cpp"
#include "surfacerzfourier.cpp"
typedef SurfaceRZFourier<PyArray> PySurfaceRZFourier;
#include "surfacexyzfourier.cpp"
typedef SurfaceXYZFourier<PyArray> PySurfaceXYZFourier;


#include "curve.cpp"
#include "pycurve.cpp"

#include "fouriercurve.cpp"
typedef FourierCurve<PyArray> PyFourierCurve;
#include "magneticaxis.cpp"
typedef StelleratorSymmetricCylindricalFourierCurve<PyArray> PyStelleratorSymmetricCylindricalFourierCurve;

#include "biot_savart.h"

namespace py = pybind11;

template <class PyFourierCurveBase = PyFourierCurve> class PyFourierCurveTrampoline : public PyCurveTrampoline<PyFourierCurveBase> {
    public:
        using PyCurveTrampoline<PyFourierCurveBase>::PyCurveTrampoline; // Inherit constructors

        int num_dofs() override {
            return PyFourierCurveBase::num_dofs();
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            PyFourierCurveBase::set_dofs_impl(_dofs);
        }

        vector<double> get_dofs() override {
            return PyFourierCurveBase::get_dofs();
        }

        void gamma_impl(PyArray& data) override {
            PyFourierCurveBase::gamma_impl(data);
        }
};

template <class PyStelleratorSymmetricCylindricalFourierCurveBase = PyStelleratorSymmetricCylindricalFourierCurve> class PyStelleratorSymmetricCylindricalFourierCurveTrampoline : public PyCurveTrampoline<PyStelleratorSymmetricCylindricalFourierCurveBase> {
    public:
        using PyCurveTrampoline<PyStelleratorSymmetricCylindricalFourierCurveBase>::PyCurveTrampoline; // Inherit constructors

        int num_dofs() override {
            return PyStelleratorSymmetricCylindricalFourierCurveBase::num_dofs();
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            PyStelleratorSymmetricCylindricalFourierCurveBase::set_dofs_impl(_dofs);
        }

        vector<double> get_dofs() override {
            return PyStelleratorSymmetricCylindricalFourierCurveBase::get_dofs();
        }

        void gamma_impl(PyArray& data) override {
            PyStelleratorSymmetricCylindricalFourierCurveBase::gamma_impl(data);
        }
};

template <class PySurfaceRZFourierBase = PySurfaceRZFourier> class PySurfaceRZFourierTrampoline : public PySurfaceRZFourierBase {
    public:
        using PySurfaceRZFourierBase::PySurfaceRZFourierBase;

        int num_dofs() override {
            return PySurfaceRZFourierBase::num_dofs();
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            PySurfaceRZFourierBase::set_dofs_impl(_dofs);
        }

        vector<double> get_dofs() override {
            return PySurfaceRZFourierBase::get_dofs();
        }

        void gamma_impl(PyArray& data) override {
            PySurfaceRZFourierBase::gamma_impl(data);
        }
};

template <class PySurfaceXYZFourierBase = PySurfaceXYZFourier> class PySurfaceXYZFourierTrampoline : public PySurfaceXYZFourierBase {
    public:
        using PySurfaceXYZFourierBase::PySurfaceXYZFourierBase;

        int num_dofs() override {
            return PySurfaceXYZFourierBase::num_dofs();
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            PySurfaceXYZFourierBase::set_dofs_impl(_dofs);
        }

        vector<double> get_dofs() override {
            return PySurfaceXYZFourierBase::get_dofs();
        }

        void gamma_impl(PyArray& data) override {
            PySurfaceXYZFourierBase::gamma_impl(data);
        }
};


PYBIND11_MODULE(simsgeopp, m) {
    xt::import_numpy();
    py::class_<PySurface, std::shared_ptr<PySurface>, PySurfaceTrampoline<PySurface>>(m, "Surface")
        .def(py::init<vector<double>,vector<double>>())
        .def("gamma", &PySurface::gamma)
        .def("gammadash1", &PySurface::gammadash1)
        .def("gammadash2", &PySurface::gammadash2)
        .def("normal", &PySurface::normal)
        .def("dnormal_by_dcoeff", &PySurface::dnormal_by_dcoeff)
        .def("surface_area", &PySurface::surface_area)
        .def("dsurface_area_by_dcoeff", &PySurface::dsurface_area_by_dcoeff)
        .def("invalidate_cache", &PySurface::invalidate_cache)
        .def("set_dofs", &PySurface::set_dofs)
        .def_readonly("quadpoints_phi", &PySurface::quadpoints_phi)
        .def_readonly("quadpoints_theta", &PySurface::quadpoints_theta);

    py::class_<PySurfaceRZFourier, std::shared_ptr<PySurfaceRZFourier>, PySurfaceRZFourierTrampoline<PySurfaceRZFourier>>(m, "SurfaceRZFourier")
        .def(py::init<int, int, int, bool, vector<double>,vector<double>>())
        .def_readwrite("rc", &PySurfaceRZFourier::rc)
        .def_readwrite("rs", &PySurfaceRZFourier::rs)
        .def_readwrite("zc", &PySurfaceRZFourier::zc)
        .def_readwrite("zs", &PySurfaceRZFourier::zs)
        .def("invalidate_cache", &PySurfaceRZFourier::invalidate_cache)
        .def("get_dofs", &PySurfaceRZFourier::get_dofs)
        .def("set_dofs", &PySurfaceRZFourier::set_dofs)
        .def("gamma", &PySurfaceRZFourier::gamma)
        .def("gammadash1", &PySurfaceRZFourier::gammadash1)
        .def("gammadash2", &PySurfaceRZFourier::gammadash2)
        .def("dgamma_by_dcoeff", &PySurfaceRZFourier::dgamma_by_dcoeff)
        .def("dgammadash1_by_dcoeff", &PySurfaceRZFourier::dgammadash1_by_dcoeff)
        .def("dgammadash2_by_dcoeff", &PySurfaceRZFourier::dgammadash2_by_dcoeff)
        .def("normal", &PySurfaceRZFourier::normal)
        .def("dnormal_by_dcoeff", &PySurfaceRZFourier::dnormal_by_dcoeff)
        .def("surface_area", &PySurfaceRZFourier::surface_area)
        .def("dsurface_area_by_dcoeff", &PySurfaceRZFourier::dsurface_area_by_dcoeff);

    py::class_<PySurfaceXYZFourier, std::shared_ptr<PySurfaceXYZFourier>, PySurfaceXYZFourierTrampoline<PySurfaceXYZFourier>>(m, "SurfaceXYZFourier")
        .def(py::init<int, int, int, bool, vector<double>,vector<double>>())
        .def_readwrite("xc", &PySurfaceXYZFourier::xc)
        .def_readwrite("xs", &PySurfaceXYZFourier::xs)
        .def_readwrite("yc", &PySurfaceXYZFourier::yc)
        .def_readwrite("ys", &PySurfaceXYZFourier::ys)
        .def_readwrite("zc", &PySurfaceXYZFourier::zc)
        .def_readwrite("zs", &PySurfaceXYZFourier::zs)
        .def("invalidate_cache", &PySurfaceXYZFourier::invalidate_cache)
        .def("get_dofs", &PySurfaceXYZFourier::get_dofs)
        .def("set_dofs", &PySurfaceXYZFourier::set_dofs)
        .def("gamma", &PySurfaceXYZFourier::gamma)
        .def("gammadash1", &PySurfaceXYZFourier::gammadash1)
        .def("gammadash2", &PySurfaceXYZFourier::gammadash2)
        .def("dgamma_by_dcoeff", &PySurfaceXYZFourier::dgamma_by_dcoeff)
        .def("dgammadash1_by_dcoeff", &PySurfaceXYZFourier::dgammadash1_by_dcoeff)
        .def("dgammadash2_by_dcoeff", &PySurfaceXYZFourier::dgammadash2_by_dcoeff)
        .def("normal", &PySurfaceXYZFourier::normal)
        .def("dnormal_by_dcoeff", &PySurfaceXYZFourier::dnormal_by_dcoeff)
        .def("surface_area", &PySurfaceXYZFourier::surface_area)
        .def("dsurface_area_by_dcoeff", &PySurfaceXYZFourier::dsurface_area_by_dcoeff);


    py::class_<PyCurve, std::shared_ptr<PyCurve>, PyCurveTrampoline<PyCurve>>(m, "Curve")
        .def(py::init<vector<double>>())
        .def("gamma", &PyCurve::gamma)
        .def("gammadash", &PyCurve::gammadash)
        .def("gammadashdash", &PyCurve::gammadashdash)
        .def("gammadashdashdash", &PyCurve::gammadashdashdash)
        .def("dgamma_by_dcoeff", &PyCurve::dgamma_by_dcoeff)
        .def("dgammadash_by_dcoeff", &PyCurve::dgammadash_by_dcoeff)
        .def("dgammadashdash_by_dcoeff", &PyCurve::dgammadashdash_by_dcoeff)
        .def("dgammadashdashdash_by_dcoeff", &PyCurve::dgammadashdashdash_by_dcoeff)
        .def("incremental_arclength", &PyCurve::incremental_arclength)
        .def("dincremental_arclength_by_dcoeff", &PyCurve::dincremental_arclength_by_dcoeff)
        .def("kappa", &PyCurve::kappa)
        .def("dkappa_by_dcoeff", &PyCurve::dkappa_by_dcoeff)
        .def("torsion", &PyCurve::torsion)
        .def("dtorsion_by_dcoeff", &PyCurve::dtorsion_by_dcoeff)
        .def("invalidate_cache", &PyFourierCurve::invalidate_cache)
        .def("set_dofs", &PyFourierCurve::set_dofs)
        .def_readonly("quadpoints", &PyCurve::quadpoints);


    py::class_<PyFourierCurve, std::shared_ptr<PyFourierCurve>, PyFourierCurveTrampoline<PyFourierCurve>>(m, "FourierCurve")
        //.def(py::init<int, int>())
        .def(py::init<vector<double>, int>())
        .def("gamma", &PyFourierCurve::gamma)
        .def("dgamma_by_dcoeff", &PyFourierCurve::dgamma_by_dcoeff)
        .def("dgamma_by_dcoeff_vjp", &PyFourierCurve::dgamma_by_dcoeff_vjp)

        .def("gammadash", &PyFourierCurve::gammadash)
        .def("dgammadash_by_dcoeff", &PyFourierCurve::dgammadash_by_dcoeff)
        .def("dgammadash_by_dcoeff_vjp", &PyFourierCurve::dgammadash_by_dcoeff_vjp)

        .def("gammadashdash", &PyFourierCurve::gammadashdash)
        .def("dgammadashdash_by_dcoeff", &PyFourierCurve::dgammadashdash_by_dcoeff)
        .def("dgammadashdash_by_dcoeff_vjp", &PyFourierCurve::dgammadashdash_by_dcoeff_vjp)

        .def("gammadashdashdash", &PyFourierCurve::gammadashdashdash)
        .def("dgammadashdashdash_by_dcoeff", &PyFourierCurve::dgammadashdashdash_by_dcoeff)
        .def("dgammadashdashdash_by_dcoeff_vjp", &PyFourierCurve::dgammadashdashdash_by_dcoeff_vjp)

        .def("incremental_arclength", &PyFourierCurve::incremental_arclength)
        .def("dincremental_arclength_by_dcoeff", &PyFourierCurve::dincremental_arclength_by_dcoeff)
        .def("kappa", &PyFourierCurve::kappa)
        .def("dkappa_by_dcoeff", &PyFourierCurve::dkappa_by_dcoeff)
        .def("torsion", &PyFourierCurve::torsion)
        .def("dtorsion_by_dcoeff", &PyFourierCurve::dtorsion_by_dcoeff)

        .def("get_dofs", &PyFourierCurve::get_dofs)
        .def("set_dofs", &PyFourierCurve::set_dofs)
        .def("num_dofs", &PyFourierCurve::num_dofs)
        .def("invalidate_cache", &PyFourierCurve::invalidate_cache)
        .def_readonly("dofs", &PyFourierCurve::dofs)
        .def_readonly("quadpoints", &PyFourierCurve::quadpoints);

    py::class_<PyStelleratorSymmetricCylindricalFourierCurve, std::shared_ptr<PyStelleratorSymmetricCylindricalFourierCurve>, PyStelleratorSymmetricCylindricalFourierCurveTrampoline<PyStelleratorSymmetricCylindricalFourierCurve>>(m, "StelleratorSymmetricCylindricalFourierCurve")
        //.def(py::init<int, int>())
        .def(py::init<vector<double>, int, int>())
        .def("gamma", &PyStelleratorSymmetricCylindricalFourierCurve::gamma)
        .def("dgamma_by_dcoeff", &PyStelleratorSymmetricCylindricalFourierCurve::dgamma_by_dcoeff)
        .def("dgamma_by_dcoeff_vjp", &PyStelleratorSymmetricCylindricalFourierCurve::dgamma_by_dcoeff_vjp)

        .def("gammadash", &PyStelleratorSymmetricCylindricalFourierCurve::gammadash)
        .def("dgammadash_by_dcoeff", &PyStelleratorSymmetricCylindricalFourierCurve::dgammadash_by_dcoeff)
        .def("dgammadash_by_dcoeff_vjp", &PyStelleratorSymmetricCylindricalFourierCurve::dgammadash_by_dcoeff_vjp)

        .def("gammadashdash", &PyStelleratorSymmetricCylindricalFourierCurve::gammadashdash)
        .def("dgammadashdash_by_dcoeff", &PyStelleratorSymmetricCylindricalFourierCurve::dgammadashdash_by_dcoeff)
        .def("dgammadashdash_by_dcoeff_vjp", &PyStelleratorSymmetricCylindricalFourierCurve::dgammadashdash_by_dcoeff_vjp)

        .def("gammadashdashdash", &PyStelleratorSymmetricCylindricalFourierCurve::gammadashdashdash)
        .def("dgammadashdashdash_by_dcoeff", &PyStelleratorSymmetricCylindricalFourierCurve::dgammadashdashdash_by_dcoeff)
        .def("dgammadashdashdash_by_dcoeff_vjp", &PyStelleratorSymmetricCylindricalFourierCurve::dgammadashdashdash_by_dcoeff_vjp)

        .def("incremental_arclength", &PyStelleratorSymmetricCylindricalFourierCurve::incremental_arclength)
        .def("dincremental_arclength_by_dcoeff", &PyStelleratorSymmetricCylindricalFourierCurve::dincremental_arclength_by_dcoeff)
        .def("kappa", &PyStelleratorSymmetricCylindricalFourierCurve::kappa)
        .def("dkappa_by_dcoeff", &PyStelleratorSymmetricCylindricalFourierCurve::dkappa_by_dcoeff)
        .def("torsion", &PyStelleratorSymmetricCylindricalFourierCurve::torsion)
        .def("dtorsion_by_dcoeff", &PyStelleratorSymmetricCylindricalFourierCurve::dtorsion_by_dcoeff)

        .def("get_dofs", &PyStelleratorSymmetricCylindricalFourierCurve::get_dofs)
        .def("set_dofs", &PyStelleratorSymmetricCylindricalFourierCurve::set_dofs)
        .def("num_dofs", &PyStelleratorSymmetricCylindricalFourierCurve::num_dofs)
        .def("invalidate_cache", &PyStelleratorSymmetricCylindricalFourierCurve::invalidate_cache)
        .def_readonly("dofs", &PyStelleratorSymmetricCylindricalFourierCurve::dofs)
        .def_readonly("quadpoints", &PyStelleratorSymmetricCylindricalFourierCurve::quadpoints)
        .def_property_readonly("nfp", &PyStelleratorSymmetricCylindricalFourierCurve::get_nfp);

    m.def("biot_savart", &biot_savart);
    m.def("biot_savart_by_dcoilcoeff_all_vjp_full", &biot_savart_by_dcoilcoeff_all_vjp_full);


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
