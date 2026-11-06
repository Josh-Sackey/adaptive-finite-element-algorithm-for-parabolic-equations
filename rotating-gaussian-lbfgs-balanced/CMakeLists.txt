cmake_minimum_required(VERSION 3.18)
project(rotating_gaussian_lbfgs_balanced LANGUAGES CXX)

find_package(deal.II 9.7.1 REQUIRED HINTS /usr/lib/x86_64-linux-gnu)
find_package(Torch REQUIRED
  PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch
    ${CMAKE_CURRENT_SOURCE_DIR}/../rotating-gaussian-classical/third_party/libtorch
  NO_DEFAULT_PATH)

deal_ii_initialize_cached_variables()

add_executable(rotating-gaussian-lbfgs-balanced
  transfer-strategy-comparison.cc
  neural_surrogate.cc)
deal_ii_setup_target(rotating-gaussian-lbfgs-balanced)
target_link_libraries(rotating-gaussian-lbfgs-balanced ${TORCH_LIBRARIES})
target_compile_options(rotating-gaussian-lbfgs-balanced PRIVATE ${TORCH_CXX_FLAGS} -O1 -g0)
set_property(TARGET rotating-gaussian-lbfgs-balanced PROPERTY CXX_STANDARD 17)
set_property(TARGET rotating-gaussian-lbfgs-balanced PROPERTY CXX_STANDARD_REQUIRED ON)
