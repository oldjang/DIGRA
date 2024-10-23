cmake_minimum_required(VERSION 3.22)
project(RangeHNSW)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET( CMAKE_CXX_FLAGS  "-O3 -march=native -lrt -DHAVE_CXX0X -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 -mavx -mavx512f" )


option(ENABLE_SSE "Enable SSE support" ON)
add_executable(RangeHNSW main.cpp hnswlib/bruteforce.h hnswlib/hnswalg.h hnswlib/hnswlib.h hnswlib/space_ip.h hnswlib/space_l2.h hnswlib/stop_condition.h hnswlib/visited_list_pool.h RangeHNSW.hpp
        utils.hpp
        DataMaker.hpp
        HNSW.hpp
        BTreeHNSW.hpp
        CompressedHNSW.hpp
        CompressedBTreeHNSW.hpp
        TreeHNSW.hpp
        BTHNSW.hpp
)
