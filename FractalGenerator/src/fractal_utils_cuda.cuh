#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "ofPixels.h"

__global__ void gpuIterateColorsKernel(double* pixels_clr,
	double pan_x,
	double pan_y,
	double mag_factor,
	int max_it,
	double tol,
	glm::vec2 size,
	glm::vec3 clr_scale,
	glm::vec3 clr_enhance);