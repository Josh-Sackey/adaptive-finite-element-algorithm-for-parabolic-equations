cmake_minimum_required(VERSION 3.18)
project(mixed_afem_nn LANGUAGES CXX)

find_package(deal.II 9.7.1 REQUIRED HINTS /usr/lib/x86_64-linux-gnu)
find_package(Torch REQUIRED
  PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch
    ${CMAKE_CURRENT_SOURCE_DIR}/../rotating-gaussian-classical/third_party/libtorch
  NO_DEFAULT_PATH)

deal_ii_initialize_cached_variables()

add_executable(mixed-afem-nn mixed-afem-nn.cc neural_surrogate.cc)
deal_ii_setup_target(mixed-afem-nn)
target_link_libraries(mixed-afem-nn ${TORCH_LIBRARIES})
target_compile_options(mixed-afem-nn PRIVATE ${TORCH_CXX_FLAGS})
set_property(TARGET mixed-afem-nn PROPERTY CXX_STANDARD 17)
set_property(TARGET mixed-afem-nn PROPERTY CXX_STANDARD_REQUIRED ON)
