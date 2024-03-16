cmake_minimum_required(VERSION 3.12)
project(BasicBlockCountPass)
set(CMAKE_CXX_STANDARD 14)
find_package(LLVM REQUIRED CONFIG)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

include_directories(include)
add_subdirectory(src)




