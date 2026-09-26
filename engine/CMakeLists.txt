cmake_minimum_required(VERSION 3.16)
project(stammtisch_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
option(JASS_NATIVE "Tune for the build machine (-march=native)" ON)

add_compile_options(-Wall -Wextra)
if(JASS_NATIVE)
  add_compile_options(-march=native)
endif()

add_library(jass_core STATIC src/rules.cpp src/agents.cpp)
target_include_directories(jass_core PUBLIC include)
set_target_properties(jass_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(jass SHARED src/capi.cpp)
target_link_libraries(jass PRIVATE jass_core)

find_package(Threads REQUIRED)
add_executable(jass-sim src/sim_main.cpp)
target_link_libraries(jass-sim PRIVATE jass_core Threads::Threads)
