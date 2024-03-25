cmake_minimum_required(VERSION 3.12)
# 设置库文件的输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_library(BasicBlockCountPass
  MODULE
  BasicBlockCountPass.cpp
  OutputHandler.cpp
)
# 这里假设LLVM库已经在根目录的CMakeLists.txt中找到和链接
# 可能需要根据实际情况链接到特定的LLVM库
target_link_libraries(BasicBlockCountPass
    ${LLVM_LIBRARIES}
)

# 设置C++标准
set_property(TARGET BasicBlockCountPass PROPERTY CXX_STANDARD 14)