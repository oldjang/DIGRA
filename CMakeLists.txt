cmake_minimum_required(VERSION 3.22)
project(Digra)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET( CMAKE_CXX_FLAGS  "-O3 -march=native -lrt -DHAVE_CXX0X -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 -mavx -mavx512f" )


option(ENABLE_SSE "Enable SSE support" ON)
add_executable(Digra main.cpp hnswlib/bruteforce.h hnswlib/hnswalg.h hnswlib/hnswlib.h hnswlib/space_ip.h hnswlib/space_l2.h hnswlib/stop_condition.h hnswlib/visited_list_pool.h 
        utils.hpp
        DataMaker.hpp
        TreeHNSW.hpp
)
