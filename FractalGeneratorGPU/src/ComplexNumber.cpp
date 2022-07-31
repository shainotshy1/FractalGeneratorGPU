#include "ComplexNumber.h"


ComplexNumber::ComplexNumber() 
	: ComplexNumber(0.0, 0.0)
{
}

ComplexNumber::ComplexNumber(double a, double b)
	: a_{a}, b_{b}
{
}

ComplexNumber::~ComplexNumber()
{
}

double ComplexNumber::get_sqr_mag()
{
	return a_ * a_ + b_ * b_;
}

std::ostream& operator<<(std::ostream& os, const ComplexNumber& c)
{
	os << c.a_ << " + " << c.b_ << "i";

	return os;
}

ComplexNumber operator*(const ComplexNumber& c1, const ComplexNumber& c2)
{
	double a{ c1.a_ * c2.a_ - c1.b_ * c2.b_ };
	double b{ c1.a_ * c2.b_ + c1.b_ * c2.a_ };

	return ComplexNumber(a, b);
}
