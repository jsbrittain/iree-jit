cmake_minimum_required(VERSION 3.21...3.24)

#-------------------------------------------------------------------------------
# Project configuration
#-------------------------------------------------------------------------------

project(iree-jit
  VERSION 0.1
  DESCRIPTION "IREE based JIT implementation in C++"
  LANGUAGES CXX C
)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#-------------------------------------------------------------------------------
# IREE subproject configuration
#-------------------------------------------------------------------------------

set(IREE_BUILD_COMPILER ON)
#set(IREE_BUILD_TESTS OFF)
#set(IREE_BUILD_SAMPLES OFF)

add_subdirectory(third_party/iree EXCLUDE_FROM_ALL)

#-------------------------------------------------------------------------------
# Application
#-------------------------------------------------------------------------------

add_subdirectory(src)
