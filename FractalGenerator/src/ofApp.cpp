#include "ofApp.h"
#include "device_info.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//--------------------------------------------------------------
void ofApp::setup() {

	NUM_THREADS = std::thread::hardware_concurrency();

	cuda_status_ = cudaSetDevice(0);
	GPU_ENABLED = cuda_status_ == cudaSuccess;

	double scale = 1;
	int w = ofGetScreenHeight() * scale;
	int h = ofGetScreenHeight() * scale;
	ofSetWindowShape(w, h);
	ofSetWindowPosition(ofGetScreenWidth() / 2 - w / 2,
		ofGetScreenHeight() / 2 - h / 2);

	mouse_pos_.x = ofGetWindowWidth() / 2;
	mouse_pos_.y = ofGetWindowHeight() / 2;

	input_keys_ = {
		{ArrowLeft, false},
		{ArrowRight, false},
		{ArrowUp, false},
		{ArrowDown, false},
		{KEY_Space, false},
		{KEY_R, false},
		{KEY_G, false}
	};

	fractal_generator_ = new FractalGenerator(GPU_ENABLED,
		NUM_THREADS,
		500,  //quality in pixels _x_
		150); //initial max iterations
}
//--------------------------------------------------------------

void ofApp::exit()
{
	delete fractal_generator_;

	if (GPU_ENABLED) {
		cuda_status_ = cudaDeviceReset();
		if (cuda_status_ != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return;
		}
	}
}

//--------------------------------------------------------------
void ofApp::update() {
	resetWindowDimensions();

	processInput();

	fractal_generator_->run();
}

//--------------------------------------------------------------
void ofApp::draw() {
	fractal_generator_->display();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	auto it = input_keys_.find(key);
	if (it == input_keys_.end()) {
		input_keys_.try_emplace(key, true);
	}
	else {
		it->second = true;
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {
	auto it = input_keys_.find(key);
	it->second = false;

	if (key == KEY_P) {
		fractal_generator_->save();
	}

	if (key == KEY_G) {
		fractal_generator_->toggleGPU();
	}
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y) {
	mouse_pos_.x = x;
	mouse_pos_.y = y;
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::resetWindowDimensions()
{
	int w = ofGetWindowWidth();
	int h = ofGetWindowHeight();
	int dim = w > h ? w : h;
	ofSetWindowShape(dim, dim);
}

void ofApp::processInput()
{
	//determine zoom direction
	int zoom_dir = 0;
	if (input_keys_.find(KEY_Space)->second) {
		zoom_dir = 1;
	}
	else if (input_keys_.find(KEY_R)->second) {
		zoom_dir = -1;
	}

	//determine shift direction
	glm::vec2 shift_dir = glm::vec2(0, 0);
	if (input_keys_.find(ArrowRight)->second) {
		shift_dir.x = 1;
	}
	else if (input_keys_.find(ArrowLeft)->second) {
		shift_dir.x = -1;
	}
	if (input_keys_.find(ArrowUp)->second) {
		shift_dir.y = 1;
	}
	else if (input_keys_.find(ArrowDown)->second) {
		shift_dir.y = -1;
	}

	//zoom and shift fractal display
	fractal_generator_->zoom(zoom_dir);
	fractal_generator_->shift(shift_dir);
}
