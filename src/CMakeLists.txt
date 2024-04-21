cmake_minimum_required(VERSION 3.12)
# 设置库文件的输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_library(IRControlFlowFuzz
  MODULE
  Utils.cpp
  LegacyIndirectBrExpand.cpp
  LegacyLowerSwitch.cpp
  FlatPlus.cpp
  BogusControlFlow.cpp

  BasicBlockCountPass.cpp
  OutputHandler.cpp
  AsmInsertionPassFuzz.cpp
)

add_dependencies(IRControlFlowFuzz intrinsics_gen)
# 设置C++标准
set_property(TARGET IRControlFlowFuzz PROPERTY CXX_STANDARD 14)


