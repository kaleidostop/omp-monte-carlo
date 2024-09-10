cmake_minimum_required(VERSION 3.17)
project(omp_montecarlo)

set(CMAKE_CXX_STANDARD 17)


OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
    FIND_PACKAGE(OpenMP)
    IF(OPENMP_FOUND)
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ENDIF()
ENDIF()


add_executable(omp_montecarlo hit.cpp main.cpp)
#add_executable(testing test/test.cpp hit.cpp)
