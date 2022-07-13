#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#define gpuErrchk(ans){gpuAssert((ans),__FILE__,__LINE__);}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

template <typename T>
void deleteOnDevice(T* v)
{
	gpuErrchk(cudaFree(v));
}

template <typename T1, typename T2>
void allocateVectorOnDevice(T1 n, T2** v)
{
	gpuErrchk(cudaMalloc((void**)v, sizeof(T2) * n));
}

template <typename T>
void copyVectorToHost(int n, const T* src, T* dst)
{
	gpuErrchk(cudaMemcpy(dst, src, sizeof(T) * n, cudaMemcpyDeviceToHost));
}
