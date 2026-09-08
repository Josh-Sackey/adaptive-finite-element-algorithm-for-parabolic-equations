cmake_minimum_required(VERSION 3.13.4)

project(step_26)

find_package(deal.II 9.7.1 REQUIRED
  HINTS /usr/lib/x86_64-linux-gnu
)

deal_ii_initialize_cached_variables()

add_executable(step-26 step-26.cc)

deal_ii_setup_target(step-26)
