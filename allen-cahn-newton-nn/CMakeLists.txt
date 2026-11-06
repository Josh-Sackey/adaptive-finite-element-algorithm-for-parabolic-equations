cmake_minimum_required(VERSION 3.18)
project(allen_cahn_newton_nn LANGUAGES CXX)

find_package(deal.II 9.7.1 REQUIRED
  HINTS ${DEAL_II_DIR} /opt/dealii /usr/lib/x86_64-linux-gnu)
find_package(Torch REQUIRED
  HINTS
    ${Torch_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch
    ${CMAKE_CURRENT_SOURCE_DIR}/../rotating-gaussian-classical/third_party/libtorch)

deal_ii_initialize_cached_variables()

add_executable(allen-cahn-newton-nn
  allen-cahn-newton-nn.cc
  neural_surrogate.cc)
deal_ii_setup_target(allen-cahn-newton-nn)
target_link_libraries(allen-cahn-newton-nn ${TORCH_LIBRARIES})
target_compile_options(allen-cahn-newton-nn PRIVATE ${TORCH_CXX_FLAGS} -O1 -g0)
set_property(TARGET allen-cahn-newton-nn PROPERTY CXX_STANDARD 17)
set_property(TARGET allen-cahn-newton-nn PROPERTY CXX_STANDARD_REQUIRED ON)

install(TARGETS allen-cahn-newton-nn RUNTIME DESTINATION bin)

