#include "FractalGenerator.h"
#include "fractal_utils.h"
#include "fractal_utils_cuda.cuh"
#include "cuda_memory_utils.cuh"
#include <cmath>

#include <windows.h>
#include <Lmcons.h>

FractalGenerator::FractalGenerator(bool gpu_enabled,
	int num_threads,
	int quality,
	double max_it,
	int tol,
	double scale,
	double zoom_scale,
	double shift_speed)
	: quality_(quality),
	scale_(scale),
	max_it_(max_it),
	tol_(tol),
	zoom_scale_(zoom_scale),
	shift_speed_(shift_speed),
	num_threads_(num_threads),
	valid_gpu_(gpu_enabled),
	gpu_enabled_(gpu_enabled)
{
	int w = 400;
	int h = 50;
	iter_multiplier_ = 1;

	qual_slider_.setup("Quality", quality_, 100, 2500, w, h);
	iter_slider_.setup("Iterations", iter_multiplier_, 0, 10, w, h);
	
	color_filter_.setup("Color Filter", 255, 0, 255, w / 2, h);
	color_filter_.setDefaultHeight(h);
	color_filter_.setDefaultWidth(w / 2);

	color_enhancer_.setup("Color Enhance", 255, 0, 255, w / 2, h);
	color_enhancer_.setDefaultHeight(h);
	color_enhancer_.setDefaultWidth(w / 2);

	font1_size_ = 30;
	qual_slider_.loadFont("font1.ttf", font1_size_);
	iter_slider_.loadFont("font1.ttf", font1_size_);
	color_filter_.loadFont("font1.ttf", 20);
	color_enhancer_.loadFont("font1.ttf", 20);

	qual_slider_.setFillColor(ofColor(50, 50, 50));
	iter_slider_.setFillColor(ofColor(50, 50, 50));
	color_filter_.setFillColor(ofColor(50, 50, 50));
	color_enhancer_.setFillColor(ofColor(50, 50, 50));

	ofColor background(150,150,150);

	qual_slider_.setBackgroundColor(background);
	iter_slider_.setBackgroundColor(background);
	color_filter_.setBackgroundColor(background);
	color_enhancer_.setBackgroundColor(background);
	qual_slider_.setDefaultTextPadding(10);

	zoom_scale0_ = 1;
	x_shift_ = 0.0;
	y_shift_ = 0.0;

	color_filter_ = ofColor(220, 50, 220);
	color_enhancer_ = ofColor(148.75, 148.75, 148.75);
	
	font1_.load("font1.ttf", font1_size_);

	setQuality(quality_);
}

FractalGenerator::~FractalGenerator()
{
	if (gpu_enabled_) {
		free(pixels_clr_h_);
		deleteOnDevice(pixels_clr_d_);
	}
}

void FractalGenerator::run()
{
	if ((double)qual_slider_ != quality_) {
		if (gpu_enabled_) {
			free(pixels_clr_h_);
			deleteOnDevice(pixels_clr_d_);
		}
		setQuality((double)qual_slider_);
	}

	ofColor clr_picked = (ofColor)color_filter_;
	clr_ = glm::vec3(clr_picked.r / 255.0, clr_picked.g / 255.0, clr_picked.b / 255.0);

	ofColor clr_enhanced = (ofColor)color_enhancer_;
	glm::vec3 min_val(-3, -3, -3);
	glm::vec3 max_val(3, 3, 3);
	double r = (clr_enhanced.r / 255.0) * (max_val.x - min_val.x) + min_val.x;
	double g = (clr_enhanced.g / 255.0) * (max_val.y - min_val.y) + min_val.y;
	double b = (clr_enhanced.b / 255.0) * (max_val.z - min_val.z) + min_val.z;
	clr_enhance_ = glm::vec3(r, g, b);
	
	iter_multiplier_ = (double)iter_slider_;

	pan_x_ = 2.5 / scale_ + x_shift_;
	pan_y_ = 2 / scale_ + y_shift_;

	generateFractal();
}

void FractalGenerator::zoom(int dir)
{
	if (dir == 0) return;
	double it_shift = 1.03;
	double shift = exp(zoom_scale_ - zoom_scale0_) / quality_;
	zoom_scale0_ = zoom_scale_;

	if (dir < 0) {
		scale_ /= zoom_scale_;
		zoom_scale_ -= shift;
		max_it_ /= it_shift;
	}
	else {
		scale_ *= zoom_scale_;
		zoom_scale_ += shift;
		max_it_ *= it_shift;
	}
}

void FractalGenerator::shift(const glm::vec2 dir)
{
	double shift = exp(zoom_scale_ / zoom_scale0_) / quality_;

	x_shift_ += dir.x * ((shift_speed_ + shift * shift) / scale_);
	y_shift_ += dir.y * ((shift_speed_ + shift * shift) / scale_);
}

void FractalGenerator::display()
{
	drawFractal();
	displayInfo();
	displayGUI();
}


void FractalGenerator::drawFractal() const
{
	img_.draw(0, 0, ofGetWindowWidth(), ofGetWindowHeight());
}

void FractalGenerator::displayGUI()
{
	int w = qual_slider_.getWidth();
	int h = qual_slider_.getHeight();
	int slider_dist = 25;

	glm::vec2 pos(ofGetWindowWidth() - 100 - w, 200 - h);

	qual_slider_.setPosition(pos.x, pos.y);
	iter_slider_.setPosition(pos.x, pos.y + (slider_dist + h));

	color_filter_.setPosition(pos.x, pos.y + (slider_dist + h) * 2);
	color_enhancer_.setPosition(pos.x + w / 2, pos.y + (slider_dist + h) * 2);

	qual_slider_.draw();
	iter_slider_.draw();

	color_filter_.draw();
	color_enhancer_.draw();
}

void FractalGenerator::displayInfo() const
{
	int x = 100;
	int y = 200;
	int precision = 6;

	string gpu_enabled = gpu_enabled_ ? "Yes" : "No";

	string scale_str = "Mag: " + ofToString(scale_, precision) + " x";
	string coord_str = "Coord: " + ofToString(pan_x_, precision) + " + "
		+ ofToString(pan_y_, precision) + " i";
	string it_str = "Iterations: " + ofToString(max_it_*iter_multiplier_);
	string fps_str = "FPS: " + ofToString(ofGetFrameRate(), 2);
	string gpu_str = "GPU Enabled: " + ofToString(gpu_enabled);

	vector<string> info{
		scale_str,
		coord_str,
		it_str,
		fps_str,
		gpu_str
	};

	for (int i = 0; i < info.size(); i++) {
		font1_.drawString(info.at(i), x, y + font1_size_ * 1.5 * i);
	}
}

void FractalGenerator::setPixelColor(const int i, const int j, const ofColor clr)
{
	img_.setColor(i, j, clr);
}

void FractalGenerator::setBackground(const ofColor clr)
{
	img_.setColor(clr);
}

void FractalGenerator::generateFractal()
{
	uint64_t num_pixels = size_.x * size_.y;
	uint64_t step = num_pixels / num_threads_;
	std::vector<std::thread> threads;
	ofPixels pixels = img_.getPixels();

	int max_it = max_it_ * iter_multiplier_;

	if (gpu_enabled_) {
		gpuIterateColors(pan_x_,
			pan_y_,
			num_pixels,
			scale_ * quality_ / 4,
			max_it,
			tol_,
			size_,
			clr_,
			clr_enhance_,
			pixels_clr_d_);

		copyVectorToHost(num_pixels * 3, pixels_clr_d_, pixels_clr_h_);

		for (int i = 0; i < num_pixels; i++) {

			int w = size_.x;
			int pix_i = i / w;
			int pix_j = i % w;

			double r = pixels_clr_h_[i * 3];
			double g = pixels_clr_h_[i * 3 + 1];
			double b = pixels_clr_h_[i * 3 + 2];
			double a = 255;

			ofColor pix_clr(r, g, b, a);
			pixels.setColor(pix_i, pix_j, pix_clr);
		}
	}
	else {
		for (int i = 0; i < num_threads_; i++) {
			threads.push_back(std::thread(cpuIterateColors,
				std::ref(pixels),
				pan_x_,
				pan_y_,
				i * step,
				(i + 1) * step,
				scale_,
				quality_,
				max_it,
				tol_,
				size_,
				clr_,
				clr_enhance_));
		}

		for (std::thread& t : threads) {
			if (t.joinable()) {
				t.join();
			}
		}
	}

	img_.setFromPixels(pixels);
	img_.update();
}

void FractalGenerator::setQuality(int quality)
{
	quality_ = quality;
	size_ = glm::vec2(quality_, quality_);
	img_.allocate(size_.x, size_.y, OF_IMAGE_COLOR);
	setBackground(ofColor::black);

	if (gpu_enabled_) {
		int n = size_.x * size_.y * 3;
		pixels_clr_h_ = new double[n];
		allocateVectorOnDevice(n, &pixels_clr_d_);
	}
}

void FractalGenerator::save(int quality)
{
	int temp_quality = quality_;
	int temp_max_it = max_it_;

	max_it_ = max_it_ * iter_multiplier_;
	setQuality(quality);
	generateFractal();

	auto time = std::time(nullptr);
	std::stringstream ss;
	ss << std::put_time(std::localtime(&time), "%F_%T"); // ISO 8601 without timezone information.
	auto s = ss.str();
	std::replace(s.begin(), s.end(), ':', '-');
	
	boost::filesystem::path file_name;
	std::string username_str;
	TCHAR username[UNLEN + 1];
	DWORD username_len = UNLEN + 1;

	if (GetUserName((TCHAR*)username, &username_len)) {

#ifndef UNICODE
		username_str = username;
#else
		std::wstring wStr = username;
		username_str = std::string(wStr.begin(), wStr.end());
#endif
		file_name.append("C:\\Users\\" + username_str + "\\Pictures\\FractalGens\\");
	};
	file_name.append("mandelbrot_" + s + ".jpg");

	img_.save(file_name, OF_IMAGE_QUALITY_BEST);

	setQuality(temp_quality);
	max_it_ = temp_max_it;
}

void FractalGenerator::toggleGPU()
{
	gpu_enabled_ = valid_gpu_ ? !gpu_enabled_ : false;
}

void FractalGenerator::cpuIterateColors(ofPixels& pixels,
	double pan_x,
	double pan_y,
	int start_i,
	int end_i,
	double scale,
	int quality,
	int max_it,
	double tol,
	glm::vec2 size,
	glm::vec3 clr,
	glm::vec3 clr_enhance) {

	double mag_factor = scale * quality / 4.0;

	for (int i = start_i; i < end_i; i++) {

		int w = size.x;
		int pix_i = i / w;
		int pix_j = i % w;

		double x = pix_i / mag_factor - pan_x;
		double y = pix_j / mag_factor - pan_y;

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

		float r = 255 * pow(val, clr_enhance.y) * clr.x;
		float g = 255 * pow(val, clr_enhance.z) * clr.y;
		float b = 255 * pow(val, clr_enhance.x) * clr.z;
		float a = 255;

		ofColor clr = ofColor(r, g, b, a);
		pixels.setColor(pix_i, pix_j, clr);
	}
}