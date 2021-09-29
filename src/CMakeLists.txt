cmake_minimum_required(VERSION 3.17)

project(gnn_ONNX_Runtime_Inference VERSION 0.0.1 LANGUAGES CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND INFR_CXX_FLAGS -Wno-error=deprecated-declarations)
endif(CMAKE_COMPILER_IS_GNUCXX)

set(INFR_CUDA_FLAGS "")
list(APPEND INFR_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND INFR_CUDA_FLAGS -Werror=cross-execution-space-call -Wno-deprecated-declarations -Xptxas=--disable-warnings)
list(APPEND INFR_CUDA_FLAGS -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable)
list(APPEND INFR_CUDA_FLAGS -Xfatbin=-compress-all)

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    list(APPEND INFR_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif(CMAKE_BUILD_TYPE MATCHES Debug)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_C_FLAGS "-Wno-error=format-truncation") 

find_package(CUDA REQUIRED)
find_package(CUDAToolkit)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS "/global/homes/a/alazar/.local/include/onnxruntime/include/")
find_path(ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS cuda_provider_factory.h HINTS "/global/homes/a/alazar/.local/include/onnxruntime/include")
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS "/global/homes/a/alazar/.local/lib-bak/onnxruntime/gpu")

add_executable(inference inference.cpp)    
set_target_properties(inference PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(inference PRIVATE 
     ${CUDA_INCLUDE_DIRS}
     ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} 
     ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS} )
target_link_libraries(inference PRIVATE  ${CUDA_RUNTIME_LIB} ${TORCH_LIBRARIES} ${ONNX_RUNTIME_LIB} CUDA::cudart)

set_property(TARGET inference PROPERTY CXX_STANDARD 14)
set_property(TARGET inference PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET inference PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET inference PROPERTY CUDA_STANDARD 14)

target_compile_options(inference
    PUBLIC  "$<$<COMPILE_LANGUAGE:CXX>:${INFR_CXX_FLAGS}>"
            "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${INFR_CUDA_FLAGS}>>")