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
        .def("dgamma_by_dphi", &PyCurve::dgamma_by_dphi)
        .def("dgamma_by_dcoeff", &PyCurve::dgamma_by_dcoeff)
        .def("invalidate_cache", &PyFourierCurve::invalidate_cache)
        .def_readonly("quadpoints", &PyCurve::quadpoints);


    py::class_<PyFourierCurve>(m, "FourierCurve")
        .def(py::init<int, int>())
        .def("gamma", &PyFourierCurve::gamma)
        .def("dgamma_by_dphi", &PyFourierCurve::dgamma_by_dphi)
        .def("dgamma_by_dcoeff", &PyFourierCurve::dgamma_by_dcoeff)
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
