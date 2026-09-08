cmake_minimum_required(VERSION 3.18)
project(fig8_quads_nn LANGUAGES CXX)

find_package(deal.II 9.7.1 REQUIRED HINTS /usr/lib/x86_64-linux-gnu)
find_package(Torch REQUIRED
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch
  NO_DEFAULT_PATH)

deal_ii_initialize_cached_variables()

add_executable(fig8-quads-nn
  fig8-quads-nn.cc
  neural_surrogate.cc)
deal_ii_setup_target(fig8-quads-nn)
target_link_libraries(fig8-quads-nn ${TORCH_LIBRARIES})
target_compile_options(fig8-quads-nn PRIVATE ${TORCH_CXX_FLAGS})
set_property(TARGET fig8-quads-nn PROPERTY CXX_STANDARD 17)
set_property(TARGET fig8-quads-nn PROPERTY CXX_STANDARD_REQUIRED ON)
