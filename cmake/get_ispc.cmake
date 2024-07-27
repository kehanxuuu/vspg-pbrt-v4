## Copyright 2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

set(ISPC_VERSION 1.24.0 CACHE STRING "")

set(SUBPROJECT_NAME ispc-v${ISPC_VERSION})

if (APPLE)
  set(ISPC_SUFFIX "macOS.universal.tar.gz")
elseif(WIN32)
  set(ISPC_SUFFIX "windows.zip")
else()
  set(ISPC_SUFFIX "linux-oneapi.tar.gz")
endif()

set(ISPC_URL "https://github.com/ispc/ispc/releases/download/v${ISPC_VERSION}/ispc-v${ISPC_VERSION}-${ISPC_SUFFIX}" CACHE STRING "")

if(WIN32)
  set(_ISPC_SRC_LIB_NAME lib)
  set(_ISPC_DST_LIB_NAME lib)
elseif(APPLE)
  set(_ISPC_SRC_LIB_NAME lib)
  set(_ISPC_DST_LIB_NAME lib)
else()
  set(_ISPC_SRC_LIB_NAME lib64)
  set(_ISPC_DST_LIB_NAME lib64)
  # set(FIND_LIBRARY_USE_LIB64_PATHS TRUE)
endif()

ExternalProject_Add(ispc
  PREFIX ${SUBPROJECT_NAME}
  STAMP_DIR ${SUBPROJECT_NAME}/stamp
  SOURCE_DIR ${SUBPROJECT_NAME}/src
  BINARY_DIR ${SUBPROJECT_NAME}
  URL ${ISPC_URL}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND
    COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/bin ${CMAKE_INSTALL_PREFIX}/bin
    COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/${_ISPC_SRC_LIB_NAME} ${CMAKE_INSTALL_PREFIX}/${_ISPC_DST_LIB_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include ${CMAKE_INSTALL_PREFIX}/include
  BUILD_ALWAYS OFF
)

set(ISPC_EXECUTABLE "${CMAKE_INSTALL_PREFIX}/bin/ispc${CMAKE_EXECUTABLE_SUFFIX}")