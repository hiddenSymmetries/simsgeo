#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;


#include "curve.cpp"
#include "pycurve.cpp"

#include "fouriercurve.cpp"
typedef FourierCurve<PyArray> PyFourierCurve;

namespace py = pybind11;

PYBIND11_MODULE(simsgeopp, m) {
    xt::import_numpy();

    py::class_<Curve<PyArray>, std::shared_ptr<Curve<PyArray>>, PyCurve>(m, "Curve")
        .def(py::init<int>())
        .def("gamma", &PyCurve::gamma)
        .def("gammadash", &PyCurve::gammadash)
        .def("gammadashdash", &PyCurve::gammadashdash)
        .def("gammadashdashdash", &PyCurve::gammadashdashdash)
        .def("dgamma_by_dcoeff", &PyCurve::dgamma_by_dcoeff)
        .def("dgammadash_by_dcoeff", &PyCurve::dgammadash_by_dcoeff)
        .def("dgammadashdash_by_dcoeff", &PyCurve::dgammadashdash_by_dcoeff)
        .def("dgammadashdashdash_by_dcoeff", &PyCurve::dgammadashdashdash_by_dcoeff)
        //.def("kappa", &PyCurve::kappa)
        //.def("torsion", &PyCurve::torsion)
        .def("invalidate_cache", &PyFourierCurve::invalidate_cache)
        .def_readonly("quadpoints", &PyCurve::quadpoints);


    py::class_<PyFourierCurve>(m, "FourierCurve")
        .def(py::init<int, int>())
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

        //.def("kappa", &PyFourierCurve::kappa)
        //.def("torsion", &PyFourierCurve::torsion)

        .def("get_dofs", &PyFourierCurve::get_dofs)
        .def("set_dofs", &PyFourierCurve::set_dofs)
        .def("num_dofs", &PyFourierCurve::num_dofs)
        .def("invalidate_cache", &PyFourierCurve::invalidate_cache);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
