#include "ofMain.h"
#include "ofApp.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "../device_info.h"

//========================================================================
int main() {

	int w = 1024;
	int h = 768;
	int scale = 2;
	ofSetupOpenGL(w * scale, h * scale, OF_WINDOW);			// <-------- setup the GL context

	// this kicks off the running of my app
	// can be OF_WINDOW or OF_FULLSCREEN
	// pass in width and height too:
	ofRunApp(new ofApp());

	return 0;
}