/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-10-24
 */

#include "StripeGenerator.hpp"

using namespace StructureLightBase;

StripeGenerator::StripeGenerator(int width, int height)
{
	this->width = width;
	this->height = height;

	this->type = MULTI_WAVELENGTH_HETERODYNE;

	this->wavelengths.clear();
	this->wavelengths.push_back(24);
	this->wavelengths.push_back(26);
	this->wavelengths.push_back(28);

	this->phase_shift_number = 4;

	this->A = 128;
	this->B = 90;

	this->reset_index();
}

StripeGenerator::~StripeGenerator()
{
}

void StripeGenerator::reset_index()
{
	this->current_wavelength_idx = 0;
	this->current_phase_shift_idx = 0;
}

void StripeGenerator::set_type(StripeType type)
{
	this->type = type;
	this->reset_index();
}


void StripeGenerator::set_waveLengths(const std::vector<int>& wavelengths)
{
	this->wavelengths = wavelengths;
	this->reset_index();
}


void StripeGenerator::set_phase_shift_number(int N)
{
	this->phase_shift_number = N;
	this->reset_index();
}


void StripeGenerator::set_A(int A)
{
	this->A = A;
	this->reset_index();
}


void StripeGenerator::set_B(int B)
{
	this->B = B;
	this->reset_index();
}

int StripeGenerator::get_pattern_size()
{
	switch (this->type)
	{
	case MULTI_WAVELENGTH_HETERODYNE:
		return (this->phase_shift_number * this->wavelengths.size());
	case GRAY_CODE:
		return 0;
	case COMPLEMENTARY_GRAY_CODE:
		return 0;
	default:
		return 0;
	}
}

inline void generate_pattern(cv::Mat& pattern, int w, int h, int wavelength, double phase_shift, int A, int B)
{
	pattern = cv::Mat::zeros(cv::Size(w, h), CV_8UC1);

	for (int x = 0; x < w; x++)
	{
		uchar I = A + B * sin(CV_2PI * x / wavelength + phase_shift);
		pattern.col(x) = I;
	}
}

bool StripeGenerator::next(cv::Mat& pattern)
{
	if (this->current_wavelength_idx >= this->wavelengths.size())
	{
		return false;
	}

	generate_pattern(
		pattern, this->width, this->height,
		this->wavelengths[this->current_wavelength_idx],
		this->current_phase_shift_idx * CV_2PI / this->phase_shift_number,
		this->A, this->B);

	this->current_phase_shift_idx++;

	if (this->current_phase_shift_idx == this->phase_shift_number)
	{
		this->current_phase_shift_idx = 0;
		this->current_wavelength_idx++;
	}

	return true;
}