/**
 * Copyright @2023 Sharemon. All rights reserved.
 *
 @author: sharemon
 @date: 2023-10-24
 */

#include "DepthReconstrcutor.hpp"

using namespace StructureLightBase;

DepthReconstructor::DepthReconstructor(StripeGenerator* strip_generator)
{
	this->strip_generator = strip_generator;
	this->min_B = 0;
}

DepthReconstructor::~DepthReconstructor()
{
}

void DepthReconstructor::set_min_B(int B)
{
	this->min_B = B;
}

void phase_reconstruct_from_shift(const std::vector<cv::Mat>& in, cv::Mat& out, int min_B, cv::Mat& B_mask)
{
	int w = in[0].cols;
	int h = in[0].rows;

	double phase_shift = CV_2PI / in.size();

	out = cv::Mat::zeros(in[0].size(), CV_64FC1);

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			double sum_sin = 0;
			double sum_cos = 0;

			for (int i = 0; i < in.size(); i++)
			{
				sum_sin += (in[i].at<uchar>(y, x) * sin(i * phase_shift));
				sum_cos += (in[i].at<uchar>(y, x) * cos(i * phase_shift));
			}

			double B = sqrt(sum_sin * sum_sin + sum_cos * sum_cos) * 2 / in.size();
			if (B > min_B)
			{
				out.at<double>(y, x) = -atan2(sum_sin, sum_cos);
			}
			else
			{
				out.at<double>(y, x) = -CV_PI;
				B_mask.at<uchar>(y, x) = 0;
			}
		}
	}
}


double phase_diff(const cv::Mat& in1, const cv::Mat& in2, double T1, double T2, cv::Mat& out, cv::Mat& K)
{
	int w = in1.cols;
	int h = in1.rows;

	double t = T2 / (T2 - T1);

	out = cv::Mat::zeros(cv::Size(w, h), CV_64FC1);

	double max = -DBL_MAX;
	double min = DBL_MAX;
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			double phi1 = in1.at<double>(y, x);
			double phi2 = in2.at<double>(y, x);

			double delta_phi = phi1 >= phi2 ? phi1 - phi2 : CV_2PI - (phi2 - phi1);
			int k = round((t * delta_phi - phi1) / CV_2PI);
			
			out.at<double>(y, x) = k * CV_2PI + phi1;
			K.at<double>(y, x) = 5 * abs(0.5 - abs(abs((t * delta_phi - phi1) / CV_2PI - (int)((t * delta_phi - phi1) / CV_2PI)) - 0.5));

			max = (max < out.at<double>(y, x)) ? out.at<double>(y, x) : max;
			min = (min > out.at<double>(y, x)) ? out.at<double>(y, x) : min;
		}
	}

	out -= min;
	out.convertTo(out, CV_64FC1, CV_2PI / (max - min + 0.2));

	return (T1 * T2 / abs(T1 - T2));
}


void merge_multi_wavelength(const std::vector<cv::Mat>& in, const std::vector<int>& wavelengths, cv::Mat& out)
{
	if (in.size() != 3)	// process 3 multi-wavelength only now
	{
		out = cv::Mat();
		return;
	}

	cv::Mat phase_diff_result12;
	cv::Mat K12 = cv::Mat::zeros(in[0].size(), CV_64FC1);
	double wavelength12 = phase_diff(in[0], in[1], wavelengths[0], wavelengths[1], phase_diff_result12, K12);

	cv::Mat phase_diff_result23;
	cv::Mat K23 = cv::Mat::zeros(in[0].size(), CV_64FC1);
	double wavelength23 = phase_diff(in[1], in[2], wavelengths[1], wavelengths[2], phase_diff_result23, K23);

	cv::Mat K = cv::Mat::zeros(in[0].size(), CV_64FC1);
	phase_diff(phase_diff_result12, phase_diff_result23, wavelength12, wavelength23, out, K);
}


void DepthReconstructor::reconstruct(const std::vector<cv::Mat>& in, cv::Mat& out)
{
	if (in.size() != this->strip_generator->phase_shift_number * this->strip_generator->wavelengths.size())
	{
		out = cv::Mat();
		return;
	}

	std::vector<cv::Mat> phase_results;
	cv::Mat B_mask = cv::Mat(in[0].size(), CV_8UC1, 255);

	// phase shift reconstruct
	for (int i = 0; i < this->strip_generator->wavelengths.size(); i++)
	{
		std::vector<cv::Mat> image_for_same_wavelength(in.begin() + i * this->strip_generator->phase_shift_number, in.begin() + (i + 1) * this->strip_generator->phase_shift_number);
		
		cv::Mat phase_result;
		phase_reconstruct_from_shift(image_for_same_wavelength, phase_result, this->min_B, B_mask);

		//phase_result += CV_PI;

		phase_results.push_back(phase_result);
	}
	
	// multi_wavelength merge
	merge_multi_wavelength(phase_results, this->strip_generator->wavelengths, out);

	// nomalize
	out.convertTo(out, CV_64FC1, 1 / CV_2PI);

	// mask
	out.setTo(0, ~B_mask);
}
