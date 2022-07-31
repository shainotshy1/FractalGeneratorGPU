#pragma once
#include <iostream>

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
	friend ComplexNumber operator*(const ComplexNumber& c1, const ComplexNumber& c2);
	friend std::ostream& operator<<(std::ostream& os, const ComplexNumber& obj);

	//methods
	double get_sqr_mag();
};