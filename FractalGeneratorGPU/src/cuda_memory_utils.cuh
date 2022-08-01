#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#define gpu_errchk(ans){gpu_assert((ans),__FILE__,__LINE__);}

inline void gpu_assert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

template <typename T>
void delete_on_device(T* v)
{
	gpu_errchk(cudaFree(v));
}

template <typename T>
void allocate_on_device(T** v, int n = 1)
{
	gpu_errchk(cudaMalloc((void**)v, sizeof(T) * n));
}

template <typename T>
void copy_to_host(const T* src, T* dst, int n = 1)
{
	gpu_errchk(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyDeviceToHost));
}

template <typename T>
void copy_to_device(const T* src, T* dst, int n = 1) 
{
	gpu_errchk(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyHostToDevice));
}