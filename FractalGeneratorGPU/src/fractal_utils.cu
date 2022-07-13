#include "fractal_utils_cuda.cuh"
#include "fractal_utils.h"
#include "cuda_memory_utils.cuh"

void gpuIterateColors(double pan_x,
	double pan_y,
	int num_pixels,
	double mag_factor,
	int max_it,
	double tol,
	glm::vec2 size,
	glm::vec3 clr,
	glm::vec3 clr_enhance,
	double* pixels_clr_d)
{
	int n = num_pixels * 3;

	int block_size = 512;
	int num_blocks = num_pixels / block_size + 1;

	dim3 block(block_size);
	dim3 grid(num_blocks);

	gpuIterateColorsKernel << < grid, block >> > (pixels_clr_d,
		pan_x,
		pan_y,
		mag_factor,
		max_it,
		tol,
		size,
		clr,
		clr_enhance);

	cudaDeviceSynchronize();
}

__global__ void gpuIterateColorsKernel(double* pixels_clr,
	double pan_x,
	double pan_y,
	double mag_factor,
	int max_it,
	double tol,
	glm::vec2 size,
	glm::vec3 clr_scale,
	glm::vec3 clr_enhance)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int w = size.x;
	int pix_i = i / w;
	int pix_j = i % w;

	double x = pix_i / mag_factor - pan_x;
	double y = pix_j / mag_factor - pan_y;

	//check if belongs to mandelbrot

	double r_val = x; //real component of Z
	double i_val = y; //imaginary component of Z
	double val = 0;

	for (int i = 0; i < max_it; i++) {

		double temp_r_val = r_val * r_val - i_val * i_val + x;
		double temp_i_val = 2 * r_val * i_val + y;

		r_val = temp_r_val;
		i_val = temp_i_val;

		if (r_val * r_val + i_val * i_val > tol * tol) {
			val = i * 1.0 / max_it;
			break;
		}
	}

	double r = 255 * pow(val, clr_enhance.y) * clr_scale.x;
	double g = 255 * pow(val, clr_enhance.z) * clr_scale.y;
	double b = 255 * pow(val, clr_enhance.x) * clr_scale.z;
	int pixels_i = i * 3;

	pixels_clr[pixels_i] = r;
	pixels_clr[pixels_i + 1] = g;
	pixels_clr[pixels_i + 2] = b;
}