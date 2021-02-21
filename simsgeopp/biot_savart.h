#pragma once

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
typedef Eigen::Vector3d Vec3d;
typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::Tensor<double, 3> Tensor3;
typedef Eigen::Tensor<double, 4> Tensor4;
typedef Eigen::TensorMap<Tensor3> MapTensor3;
typedef Eigen::TensorMap<Tensor4> MapTensor4;
typedef Eigen::Ref<Matrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> RefMatrix;
typedef Eigen::Ref<Vector, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> RefVector;


#include <vector>
using std::vector;

#include "xsimd/xsimd.hpp"
namespace xs = xsimd;
using AlignedVector = std::vector<double, xs::aligned_allocator<double, XSIMD_DEFAULT_ALIGNMENT>>;
using SIMDVector = xs::simd_type<double>;

struct Vec3dSimd {
    SIMDVector x;
    SIMDVector y;
    SIMDVector z;

    Vec3dSimd() : x(0.), y(0.), z(0.){
    }

    Vec3dSimd(double x_, double y_, double z_) : x(x_), y(y_), z(z_){
    }

    Vec3dSimd(Vec3d xyz) : x(xyz[0]), y(xyz[1]), z(xyz[2]){
    }

    Vec3dSimd(const SIMDVector& x_, const SIMDVector& y_, const SIMDVector& z_) : x(x_), y(y_), z(z_) {
    }

    Vec3dSimd(double* xptr, double* yptr, double *zptr){
        x = xs::load_aligned(xptr);
        y = xs::load_aligned(yptr);
        z = xs::load_aligned(zptr);
    }

    void store_aligned(double* xptr, double* yptr, double *zptr){
        x.store_aligned(xptr);
        y.store_aligned(yptr);
        z.store_aligned(zptr);
    }

    SIMDVector& operator[] (int i){
        if(i==0) {
            return x;
        }else if(i==1){
            return y;
        } else{
            return z;
        }
    }

    friend Vec3dSimd operator+(Vec3dSimd lhs, const Vec3d& rhs) {
        lhs.x += rhs[0];
        lhs.y += rhs[1];
        lhs.z += rhs[2];
        return lhs;
    }

    friend Vec3dSimd operator+(Vec3dSimd lhs, const Vec3dSimd& rhs) {
        lhs.x += rhs.x;
        lhs.y += rhs.y;
        lhs.z += rhs.z;
        return lhs;
    }

    Vec3dSimd& operator+=(const Vec3dSimd& rhs) {
        this->x += rhs.x;
        this->y += rhs.y;
        this->z += rhs.z;
        return *this;
    }

    Vec3dSimd& operator-=(const Vec3dSimd& rhs) {
        this->x -= rhs.x;
        this->y -= rhs.y;
        this->z -= rhs.z;
        return *this;
    }

    friend Vec3dSimd operator-(Vec3dSimd lhs, const Vec3d& rhs) {
        lhs.x -= rhs[0];
        lhs.y -= rhs[1];
        lhs.z -= rhs[2];
        return lhs;
    }

    friend Vec3dSimd operator-(Vec3dSimd lhs, const Vec3dSimd& rhs) {
        lhs.x -= rhs.x;
        lhs.y -= rhs.y;
        lhs.z -= rhs.z;
        return lhs;
    }

    friend Vec3dSimd operator*(Vec3dSimd lhs, const SIMDVector& rhs) {
        lhs.x *= rhs;
        lhs.y *= rhs;
        lhs.z *= rhs;
        return lhs;
    }
};


inline SIMDVector inner(const Vec3dSimd& a, const Vec3dSimd& b){
    return a.x*b.x+a.y*b.y+a.z*b.z;
}

inline SIMDVector inner(const Vec3d& b, const Vec3dSimd& a){
    return a.x*b[0]+a.y*b[1]+a.z*b[2];
}

inline SIMDVector inner(const Vec3dSimd& a, const Vec3d& b){
    return a.x*b[0]+a.y*b[1]+a.z*b[2];
}

inline SIMDVector inner(int i, Vec3dSimd& a){
    if(i==0)
        return a.x;
    else if(i==1)
        return a.y;
    else
        return a.z;
}


inline double inner(const Vec3d& a, const Vec3d& b){
    return a.dot(b);
}

inline Vec3d cross(const Vec3d& a, const Vec3d& b){
    return a.cross(b);
}

inline double norm(const Vec3d& a){
    return a.norm();
}

inline Vec3dSimd cross(Vec3dSimd& a, Vec3dSimd& b){
    return Vec3dSimd(
            xsimd::fms(a.y, b.z, a.z * b.y),
            xsimd::fms(a.z, b.x, a.x * b.z),
            xsimd::fms(a.x, b.y, a.y * b.x)
            );
}

inline Vec3dSimd cross(Vec3dSimd& a, Vec3d& b){
    return Vec3dSimd(a.y * b[2] - a.z * b[1], a.z * b[0] - a.x * b[2], a.x * b[1] - a.y * b[0]);

}

inline Vec3dSimd cross(Vec3d& a, Vec3dSimd& b){
    return Vec3dSimd(a[1] * b.z - a[2] * b.y, a[2] * b.x - a[0] * b.z, a[0] * b.y - a[1] * b.x);
}

inline Vec3dSimd cross(Vec3dSimd& a, int i){
    if(i==0)
        return Vec3dSimd(SIMDVector(0.), a.z, -a.y);
    else if(i == 1)
        return Vec3dSimd(-a.z, SIMDVector(0.), a.x);
    else
        return Vec3dSimd(a.y, -a.x, SIMDVector(0.));
}

inline Vec3dSimd cross(int i, Vec3dSimd& b){
    if(i==0)
        return Vec3dSimd(SIMDVector(0.), -b.z, b.y);
    else if(i == 1)
        return Vec3dSimd(b.z, SIMDVector(0.), -b.x);
    else
        return Vec3dSimd(-b.y, b.x, SIMDVector(0.));
}

inline SIMDVector normsq(Vec3dSimd& a){
    return a.x*a.x+a.y*a.y+a.z*a.z;
}

void biot_savart(Matrix& points, vector<Matrix>& gammas, vector<Matrix>& dgamma_by_dphis, vector<RefMatrix>& B);
void biot_savart(Matrix& points, vector<Matrix>& gammas, vector<Matrix>& dgamma_by_dphis, vector<RefMatrix>& B, vector<MapTensor3>& dB_by_dX);
void biot_savart(Matrix& points, vector<Matrix>& gammas, vector<Matrix>& dgamma_by_dphis, vector<RefMatrix>& B, vector<MapTensor3>& dB_by_dX, vector<MapTensor4>& d2B_by_dXdX);



void biot_savart_by_dcoilcoeff_all_vjp_full(RefMatrix& points, vector<RefMatrix>& gammas, vector<RefMatrix>& dgamma_by_dphis, vector<double>& currents, RefMatrix& v, MapTensor3& vgrad, vector<MapTensor3>& dgamma_by_dcoeffs, vector<MapTensor3>& d2gamma_by_dphidcoeffs, vector<RefVector>& res_B, vector<RefVector>& res_dB);
