#pragma once

#include "ofMain.h"
#include "ofTexture.h"
#include "ofPixels.h"
#include "ofColor.h"
#include "ofxGui.h"

class FractalGenerator
{
private:
	//member variables

	double* pixels_clr_d_;
	double* pixels_clr_h_;

	int quality_;
	int tol_;

	double max_it_;
	double scale_;

	double zoom_scale_;
	double zoom_scale0_;
	double shift_speed_;
	double x_shift_;
	double y_shift_;

	double pan_x_;
	double pan_y_;

	bool gpu_enabled_;
	int num_threads_;

	glm::vec3 clr_;
	glm::vec3 clr_enhance_;
	glm::vec2 size_;
	ofImage img_;

	int font1_size_;
	ofTrueTypeFont font1_;
	
	ofxFloatSlider red_slider_;
	ofxFloatSlider green_slider_;
	ofxFloatSlider blue_slider_;

public:
	//constructor
	FractalGenerator(bool gpu_enabled,
		int num_threads,
		int quality = 1000,
		double max_it = 25,
		int tol = 5,
		double scale = 1.0,
		double zoom_scale = 1.01,
		double shift_speed = 0.1);

	//destructor
	~FractalGenerator();

	//methods
	void run();
	void zoom(int dir);
	void shift(const glm::vec2 dir);
	void display();
	void drawFractal() const;
	void displaySliders();
	void displayInfo() const;
	void setPixelColor(const int i, const int j, const ofColor clr);
	void setBackground(const ofColor clr);
	void generateFractal();
	void setQuality(int quality);
	void save(int quality = 5000);
	void toggleGPU();

	static void cpuIterateColors(ofPixels& pixels,
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
		glm::vec3 clr_enhance);
};