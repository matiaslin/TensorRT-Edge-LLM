# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# We prioritize TRT_PACKAGE_DIR to facilitate some x86 development systems where
# multiple TensorRT versions might exist.
if(NOT DEFINED TENSORRT_ROOT AND DEFINED TRT_PACKAGE_DIR)
  set(TENSORRT_ROOT ${TRT_PACKAGE_DIR})
endif()

# Fallback to standard system paths (useful for systems that have TensorRT
# bundled, such as JetPack, etc.)
set(TRT_HINTS ${TENSORRT_ROOT} /usr /usr/local/cuda /opt/tensorrt)

# Core TensorRT
find_path(
  TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TRT_HINTS}
  PATH_SUFFIXES
    include include/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu # Ubuntu/JetPack layout
    ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)
find_library(
  TENSORRT_LIBRARY nvinfer
  HINTS ${TRT_HINTS}
  PATH_SUFFIXES
    lib lib64 lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu # Ubuntu/JetPack layout
    ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)

# Onnx parser
find_path(ONNX_PARSER_INCLUDE_DIR NvOnnxParser.h HINTS ${TENSORRT_INCLUDE_DIR})
find_library(
  NV_ONNX_PARSER_LIB nvonnxparser
  HINTS ${TRT_HINTS}
  PATH_SUFFIXES lib lib64 ${CMAKE_SYSTEM_PROCESSOR}-linux-gnu)

# Provide a targeted error message for partial installations where the core
# library is found but the ONNX Parser is missing (consistent with legacy
# behavior)
if(TENSORRT_INCLUDE_DIR AND NOT ONNX_PARSER_INCLUDE_DIR)
  message(
    FATAL_ERROR
      "NvOnnxParser.h not found in TensorRT headers, please specify the -DONNX_PARSER_INCLUDE_DIR when invoking CMake"
  )
endif()

# Handling Onnx Parser component
set(TensorRT_OnnxParser_FOUND OFF)
if(ONNX_PARSER_INCLUDE_DIR AND NV_ONNX_PARSER_LIB)
  set(TensorRT_OnnxParser_FOUND ON)
endif()

# Standard package handling for TensorRT
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TensorRT
  REQUIRED_VARS TENSORRT_LIBRARY TENSORRT_INCLUDE_DIR
  HANDLE_COMPONENTS
  REASON_FAILURE_MESSAGE
  "TensorRT not found via auto-detection. Please specify -DTRT_PACKAGE_DIR=/path/to/TRT."
)

# Handling outputs
if(TensorRT_FOUND)
  # Only core TensorRT to allow for surgical linking
  set(TensorRT_INCLUDE_DIR ${TENSORRT_INCLUDE_DIR})
  set(TensorRT_LIBRARY ${TENSORRT_LIBRARY})

  # Core + components
  set(TensorRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
  set(TensorRT_LIBRARIES ${TENSORRT_LIBRARY})

  if(TensorRT_OnnxParser_FOUND)
    set(ONNX_PARSER_INCLUDE_DIR ${ONNX_PARSER_INCLUDE_DIR})
    set(NV_ONNX_PARSER_LIB ${NV_ONNX_PARSER_LIB})
    list(APPEND TensorRT_INCLUDE_DIRS ${ONNX_PARSER_INCLUDE_DIR})
    list(APPEND TensorRT_LIBRARIES ${NV_ONNX_PARSER_LIB})
  endif()

  list(REMOVE_DUPLICATES TensorRT_INCLUDE_DIRS)
endif()
