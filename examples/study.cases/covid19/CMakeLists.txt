cmake_minimum_required(VERSION 3.10)

project(SIR VERSION LANGUAGES CXX)
set(EXTERN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern")
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp")

find_package(Boost)

add_subdirectory(${EXTERN_DIR}/pybind11 sir_pybind11)
pybind11_add_module(
    libsir
    ${SRC_DIR}/sir.cpp
    ${SRC_DIR}/bindings.cpp)

target_link_libraries(libsir PRIVATE Boost::boost)
