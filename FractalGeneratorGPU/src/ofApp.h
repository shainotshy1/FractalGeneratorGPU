#pragma once

#include "ofMain.h"
#include "FractalGenerator.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <map>

class ofApp : public ofBaseApp {

private:
	glm::vec2 mouse_pos_;
	std::map<int, bool> input_keys_;
	FractalGenerator* fractal_generator_;
	cudaError_t cuda_status_;

public:

	enum Key
	{
		ArrowDown = 57359,
		ArrowLeft = 57358,
		ArrowRight = 57356,
		ArrowUp = 57357,
		KEY_Alt = 18,
		KEY_Backspace = 8,
		KEY_CapsLock = 20,
		KEY_Ctrl = 162,
		KEY_Delete = 46,
		KEY_End = 35,
		KEY_Enter = 13,
		KEY_Esc = 27,
		KEY_F10 = 121,
		KEY_F11 = 122,
		KEY_F12 = 123,
		NumPad0 = 96,
		NumPad1 = 97,
		NumPad2 = 98,
		NumPad3 = 99,
		NumPad4 = 100,
		NumPad5 = 101,
		NumPad6 = 102,
		NumPad7 = 103,
		NumPad8 = 104,
		NumPad9 = 105,
		KEY_PageDown = 34,
		KEY_PageUp = 33,
		KEY_Pause = 19,
		KEY_PrintScrn = 44,
		KEY_ScrollLock = 145,
		KEY_Shift = 16,
		KEY_Space = 32,
		KEY_Tab = 9,
		KEY_A = 97,
		KEY_B = 98,
		KEY_C = 99,
		KEY_D = 100,
		KEY_E = 101,
		KEY_F = 102,
		KEY_G = 103,
		KEY_H = 104,
		KEY_I = 105,
		KEY_J = 106,
		KEY_K = 107,
		KEY_L = 108,
		KEY_M = 109,
		KEY_N = 110,
		KEY_O = 111,
		KEY_P = 112,
		KEY_Q = 113,
		KEY_R = 114,
		KEY_S = 115,
		KEY_T = 116,
		KEY_U = 117,
		KEY_V = 118,
		KEY_W = 119,
		KEY_X = 120,
		KEY_Y = 121,
		KEY_Z = 122,
		KEY_Home = 36,
		KEY_Insert = 45,
		NumLock = 144,
		NumPadMinus = 109,
		NumPadStar = 106,
		NumPadPeriod = 110,
		NumPadSlash = 111,
		NumPadPlus = 107,
		KEY_BackApostrophe = 222,
		KEY_Minus = 189,
		KEY_Apostrophe = 188,
		KEY_Period = 190,
		KEY_ForwardSlash = 191,
		KEY_SemiColon = 186,
		KEY_ForwardBracket = 219,
		KEY_BackSlash = 220,
		KEY_BackBracket = 221,
		KEY_Equals = 187
	};

	void setup();
	void exit();
	void update();
	void draw();

	void keyPressed(int key);
	void keyReleased(int key);
	void mouseMoved(int x, int y);
	void mouseDragged(int x, int y, int button);
	void mousePressed(int x, int y, int button);
	void mouseReleased(int x, int y, int button);
	void mouseEntered(int x, int y);
	void mouseExited(int x, int y);
	void windowResized(int w, int h);
	void dragEvent(ofDragInfo dragInfo);
	void gotMessage(ofMessage msg);
	void resetWindowDimensions();
	void processInput();
};