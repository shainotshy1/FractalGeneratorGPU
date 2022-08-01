#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct ComplexNumber 
{
	//member variables a + bi
	double a_; //real component
	double b_; //imaginary component

	//constructor
	ComplexNumber();
	ComplexNumber(double a, double b);

	//destructor
	~ComplexNumber();

	//operator overloads
	friend std::ostream& operator<<(std::ostream& os, const ComplexNumber& obj);
	friend ComplexNumber operator*(const ComplexNumber& c1, const ComplexNumber& c2);
	void operator*=(const ComplexNumber& rhs);
	friend ComplexNumber operator+(const ComplexNumber& c1, const ComplexNumber& c2);
	void operator+=(const ComplexNumber& rhs);
	friend ComplexNumber operator-(const ComplexNumber& c1, const ComplexNumber& c2);
	void operator-=(const ComplexNumber& rhs);

	//methods
	double get_sqr_mag();
};