cmake_minimum_required(VERSION 3.18)
project(rotating_gaussian_transfer_strategy_comparison LANGUAGES CXX)

find_package(deal.II 9.7.1 REQUIRED HINTS /usr/lib/x86_64-linux-gnu)
find_package(Torch REQUIRED
  PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch
    ${CMAKE_CURRENT_SOURCE_DIR}/../rotating-gaussian-classical/third_party/libtorch
  NO_DEFAULT_PATH)

deal_ii_initialize_cached_variables()

add_executable(transfer-strategy-comparison
  transfer-strategy-comparison.cc
  neural_surrogate.cc)
deal_ii_setup_target(transfer-strategy-comparison)
target_link_libraries(transfer-strategy-comparison ${TORCH_LIBRARIES})
target_compile_options(transfer-strategy-comparison PRIVATE ${TORCH_CXX_FLAGS} -O1 -g0)
set_property(TARGET transfer-strategy-comparison PROPERTY CXX_STANDARD 17)
set_property(TARGET transfer-strategy-comparison PROPERTY CXX_STANDARD_REQUIRED ON)
