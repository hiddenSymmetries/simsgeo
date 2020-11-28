cmake_minimum_required(VERSION 3.5)
project(simsgeo)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(NumPy REQUIRED)
find_package(OpenMP)


set(CMAKE_CXX_FLAGS "-O3 -march=native")

add_subdirectory(pybind11)
set(XTENSOR_USE_XSIMD 1)
set(XTENSOR_USE_OPENMP 0)



pybind11_add_module(simsgeopp
    simsgeopp/python.cpp simsgeopp/biot_savart.cpp simsgeopp/biot_savart_derivative.cpp
    )
set_target_properties(simsgeopp
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
target_include_directories(simsgeopp PRIVATE "xtensor/include" "xtensor-python/include" "xsimd/include" "xtl/include" "blaze" "simsgeopp" ${NUMPY_INCLUDE_DIRS})
target_link_libraries(simsgeopp PUBLIC ${PYTHON_LIBRARY})


add_executable(hello_world driver/main.cpp simsgeopp/biot_savart.cpp simsgeopp/biot_savart_derivative.cpp)
set_target_properties(hello_world
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
target_include_directories(hello_world PRIVATE  "xtensor/include" "xtensor-python/include" "xsimd/include" "xtl/include" "blaze" "simsgeopp" ${PYBIND11_INCLUDE_DIR} ${NUMPY_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIR})
target_link_libraries(hello_world PUBLIC ${PYTHON_LIBRARY})


if(OpenMP_CXX_FOUND)
    target_link_libraries(simsgeopp PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(hello_world PRIVATE OpenMP::OpenMP_CXX)
endif()

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
