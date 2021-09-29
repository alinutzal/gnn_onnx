# ONNX Runtime Inference

## Introduction

ONNX Runtime C++ inference example for running ONNX GNN models on CUDA.

## Dependencies

* CMake 3.17.2
* ONNX Runtime 1.9.0 (include and library) follow instructions on [Setting up ONNX Runtime on Ubuntu 20.04 (C++ API)] (https://stackoverflow.com/questions/63420533/setting-up-onnx-runtime-on-ubuntu-20-04-c-api)
* CUDA 11.0.3 and CUDATOOLKIT

## Modules

* module load cmake/3.20.5
* module load cuda/11.0.3
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

### Build Example

```bash
$ cd gnn_onnx/src/build
$ cmake ..
$ cmake --build .

### Run Example

```bash
$ ./build/inference  --use_cuda
Inference Execution Provider: CUDA
```
