#pragma once

#include "ofPixels.h"

void gpuIterateColors(double pan_x,
	double pan_y,
	int num_pixels,
	double mag_factor,
	int max_it,
	double tol,
	glm::vec2 size,
	glm::vec3 clr,
	glm::vec3 clr_enhance,
	double* pixels_clr_d);

glm::vec3 calculateColors(double val,
	glm::vec3 clr_enhance,
	glm::vec3 clr_scale);