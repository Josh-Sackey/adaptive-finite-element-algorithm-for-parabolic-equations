cmake_minimum_required(VERSION 3.13.4)

project(fig8_quads)

find_package(deal.II 9.7.1 REQUIRED
  HINTS /usr/lib/x86_64-linux-gnu
)

# LibTorch is project-local. TorchConfig.cmake supplies include paths, ABI
# flags, libraries, and runtime search paths.
find_package(Torch REQUIRED
  PATHS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch
  NO_DEFAULT_PATH
)

deal_ii_initialize_cached_variables()

add_executable(fig8-quads fig8-quads.cc)
deal_ii_setup_target(fig8-quads)

add_executable(torch-smoke-test torch-smoke-test.cc)
target_link_libraries(torch-smoke-test PRIVATE ${TORCH_LIBRARIES})
target_compile_options(torch-smoke-test PRIVATE ${TORCH_CXX_FLAGS})
set_property(TARGET torch-smoke-test PROPERTY CXX_STANDARD 17)
set_property(TARGET torch-smoke-test PROPERTY CXX_STANDARD_REQUIRED ON)
