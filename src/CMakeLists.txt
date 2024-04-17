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
  SplitBasicBlockPass.cpp
)

add_dependencies(IRControlFlowFuzz intrinsics_gen)
# 设置C++标准
set_property(TARGET IRControlFlowFuzz PROPERTY CXX_STANDARD 14)



# add_library(BasicBlockCountPass
#   MODULE
#   BasicBlockCountPass.cpp
#   OutputHandler.cpp
# )

# add_library(AsmInsertionPass
#   MODULE
#   AsmInsertionPass.cpp
# )

# add_library(AsmInsertionPassFuzz
#   MODULE
#   AsmInsertionPassFuzz.cpp
# )

# add_library(SplitBasicBlockPass
#   MODULE
#   SplitBasicBlockPass.cpp
# )

# # 这里假设LLVM库已经在根目录的CMakeLists.txt中找到和链接
# # 可能需要根据实际情况链接到特定的LLVM库
# target_link_libraries(BasicBlockCountPass
#     ${LLVM_LIBRARIES}
# )

# target_link_libraries(AsmInsertionPass
#     ${LLVM_LIBRARIES}
# )
# target_link_libraries(AsmInsertionPassFuzz
#     ${LLVM_LIBRARIES}
# )
# target_link_libraries(SplitBasicBlockPass
#     ${LLVM_LIBRARIES}
# )
# 设置C++标准
# set_property(TARGET BasicBlockCountPass PROPERTY CXX_STANDARD 14)
# set_property(TARGET AsmInsertionPass PROPERTY CXX_STANDARD 14)
# set_property(TARGET AsmInsertionPassFuzz PROPERTY CXX_STANDARD 14)
# set_property(TARGET SplitBasicBlockPass PROPERTY CXX_STANDARD 14)


