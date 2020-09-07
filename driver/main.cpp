// Your First C++ Program

#include <iostream>

#include "fouriercurve.cpp"
typedef xt::xarray<double> CppArray;

int main() {
    auto mycurve = FourierCurve<CppArray>(10, 3);
    std::cout << mycurve.gamma()(3, 0) << std::endl;
    std::cout << mycurve.gamma()(3, 0) << std::endl;
    mycurve.invalidate_cache();
    std::cout << mycurve.gamma()(3, 0) << std::endl;
    mycurve.gamma()(3, 0) = 3.;
    std::cout << mycurve.gamma()(3, 0) << std::endl;
    std::cout << "Hello World!";
    return 0;
}
