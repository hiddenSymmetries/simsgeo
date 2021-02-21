#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;


#include "curve.cpp"
#include "pycurve.cpp"

#include "fouriercurve.cpp"
typedef FourierCurve<PyArray> PyFourierCurve;
#include "magneticaxis.cpp"
typedef StelleratorSymmetricCylindricalFourierCurve<PyArray> PyStelleratorSymmetricCylindricalFourierCurve;

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "biot_savart.h"

namespace py = pybind11;
typedef py::array_t<double, py::array::f_style | py::array::forcecast> pyarrayf;

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

PYBIND11_MODULE(simsgeopp, m) {
    xt::import_numpy();

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

    m.def("biot_savart_1", [](Matrix& points, vector<Matrix>& gammas, vector<Matrix>& dgamma_by_dphis, vector<RefMatrix>& B)
            {
            biot_savart(points, gammas, dgamma_by_dphis, B);
            });

    m.def("biot_savart_2", [](Matrix& points, vector<Matrix>& gammas, vector<Matrix>& dgamma_by_dphis, vector<RefMatrix>& B, vector<pyarrayf>& py_dB_by_dX){
                auto dB_by_dX = std::vector<MapTensor3>();
                for (int i = 0; i < py_dB_by_dX.size(); ++i) {
                    py::buffer_info buf = py_dB_by_dX[i].request();
                    if (buf.ndim != 3) {
                        throw std::runtime_error("Number of dimensions must be 3.");
                    }
                    // Without copying, represent the incoming numpy array as an Eigen::Tensor:
                    dB_by_dX.push_back(Eigen::TensorMap<Tensor3>(static_cast<double*>(buf.ptr), buf.shape[0], buf.shape[1], buf.shape[2]));

                }
                biot_savart(points, gammas, dgamma_by_dphis, B, dB_by_dX);
            });
    m.def("biot_savart_3", [](Matrix& points, vector<Matrix>& gammas, vector<Matrix>& dgamma_by_dphis, vector<RefMatrix>& B, vector<pyarrayf>& py_dB_by_dX, vector<pyarrayf>& py_d2B_by_dXdX){
                auto dB_by_dX = std::vector<MapTensor3>();
                for (int i = 0; i < py_dB_by_dX.size(); ++i) {
                    py::buffer_info buf = py_dB_by_dX[i].request();
                    if (buf.ndim != 3) {
                        throw std::runtime_error("Number of dimensions must be 3.");
                    }
                    // Without copying, represent the incoming numpy array as an Eigen::Tensor:
                    dB_by_dX.push_back(Eigen::TensorMap<Tensor3>(static_cast<double*>(buf.ptr), buf.shape[0], buf.shape[1], buf.shape[2]));

                }
                auto d2B_by_dXdX = std::vector<MapTensor4>();
                for (int i = 0; i < py_d2B_by_dXdX.size(); ++i) {
                    py::buffer_info buf = py_d2B_by_dXdX[i].request();
                    if (buf.ndim != 4) {
                        throw std::runtime_error("Number of dimensions must be 4.");
                    }
                    // Without copying, represent the incoming numpy array as an Eigen::Tensor:
                    d2B_by_dXdX.push_back(Eigen::TensorMap<Tensor4>(static_cast<double*>(buf.ptr), buf.shape[0], buf.shape[1], buf.shape[2], buf.shape[3]));

                }
                biot_savart(points, gammas, dgamma_by_dphis, B, dB_by_dX, d2B_by_dXdX);
            });
    //m.def("biot_savart_by_dcoilcoeff_all_vjp_full", &biot_savart_by_dcoilcoeff_all_vjp_full);
    m.def("biot_savart_by_dcoilcoeff_all_vjp_full", 
            [](RefMatrix& points, vector<RefMatrix>& gammas, vector<RefMatrix>& dgamma_by_dphis, vector<double>& currents, RefMatrix& v, pyarrayf& py_vgrad, vector<pyarrayf>& py_dgamma_by_dcoeffs, vector<pyarrayf>& py_d2gamma_by_dphidcoeffs, vector<RefVector>& res_B, vector<RefVector>& res_dB)
            {
                py::buffer_info buf = py_vgrad.request();
                if (buf.ndim != 3) {
                    throw std::runtime_error("Number of dimensions must be 3.");
                }
                auto vgrad = Eigen::TensorMap<Tensor3>(static_cast<double*>(buf.ptr), buf.shape[0], buf.shape[1], buf.shape[2]);

                auto dgamma_by_dcoeffs = std::vector<MapTensor3>();
                for (int i = 0; i < py_dgamma_by_dcoeffs.size(); ++i) {
                    py::buffer_info buf = py_dgamma_by_dcoeffs[i].request();
                    if (buf.ndim != 3) {
                        throw std::runtime_error("Number of dimensions must be 3.");
                    }
                    // Without copying, represent the incoming numpy array as an Eigen::Tensor:
                    dgamma_by_dcoeffs.push_back(Eigen::TensorMap<Tensor3>(static_cast<double*>(buf.ptr), buf.shape[0], buf.shape[1], buf.shape[2]));

                }

                auto d2gamma_by_dphidcoeffs = std::vector<MapTensor3>();
                for (int i = 0; i < py_d2gamma_by_dphidcoeffs.size(); ++i) {
                    py::buffer_info buf = py_d2gamma_by_dphidcoeffs[i].request();
                    if (buf.ndim != 3) {
                        throw std::runtime_error("Number of dimensions must be 3.");
                    }
                    // Without copying, represent the incoming numpy array as an Eigen::Tensor:
                    d2gamma_by_dphidcoeffs.push_back(Eigen::TensorMap<Tensor3>(static_cast<double*>(buf.ptr), buf.shape[0], buf.shape[1], buf.shape[2]));
                }
            return biot_savart_by_dcoilcoeff_all_vjp_full(points, gammas, dgamma_by_dphis, currents, v, vgrad, dgamma_by_dcoeffs, d2gamma_by_dphidcoeffs, res_B, res_dB);
            });


#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
