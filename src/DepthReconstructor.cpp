/**
 * Copyright @2023 Sharemon. All rights reserved.
 *
 @author: sharemon
 @date: 2023-10-24
 */

#include "DepthReconstructor.hpp"

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


void DepthReconstructor::set_stereo_param(StereoCommon::StereoParameter* parameter)
{
	this->stereo_param = parameter;
}


void phase_reconstruct_from_shift(const std::vector<cv::Mat>& in, cv::Mat& out, cv::Mat& B_mask,  int min_B = -1)
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
				sum_sin += (double)(in[i].at<double>(y, x) * sin(i * phase_shift));
				sum_cos += (double)(in[i].at<double>(y, x) * cos(i * phase_shift));
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


double phase_diff(const cv::Mat& in1, const cv::Mat& in2, double T1, double T2, cv::Mat& out)
{
	int w = in1.cols;
	int h = in1.rows;

	double t = T2 / (T2 - T1);

	out = cv::Mat::zeros(cv::Size(w, h), CV_64FC1);

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			double phi1 = in1.at<double>(y, x);
			double phi2 = in2.at<double>(y, x);

			double delta_phi = phi1 > phi2 ? phi1 - phi2 : CV_2PI - (phi2 - phi1);
			int k = round((t * delta_phi - phi1) / CV_2PI);

			out.at<double>(y, x) = k * CV_2PI + phi1;
		}
	}

	return (T1 * T2 / (T1 - T2));
}


void merge_multi_wavelength(const std::vector<cv::Mat>& in, const std::vector<cv::Mat>& ideal, const std::vector<int>& wavelengths, cv::Mat& out)
{
	if (in.size() != 3 && ideal.size() != 3)	// process 3 multi-wavelength only now
	{
		out = cv::Mat();
		return;
	}

	cv::Mat phase_diff_result12, phase_diff_result23;
	cv::Mat phase_diff_ideal12, phase_diff_ideal23, phase_diff_ideal123;
	double wavelength12, wavelength23, min, max;

	// calcualte phase diff and use ideal min and max to normalize
	// 12
	wavelength12 = phase_diff(ideal[0], ideal[1], wavelengths[0], wavelengths[1], phase_diff_ideal12);
	cv::minMaxLoc(phase_diff_ideal12, &min, &max);

	wavelength12 = phase_diff(in[0], in[1], wavelengths[0], wavelengths[1], phase_diff_result12);
	phase_diff_ideal12 = (phase_diff_ideal12 - min) / (max - min + 0.2) * CV_2PI;
	phase_diff_result12 = (phase_diff_result12 - min) / (max - min + 0.2) * CV_2PI;

	// 23
	wavelength23 = phase_diff(ideal[1], ideal[2], wavelengths[1], wavelengths[2], phase_diff_ideal23);
	cv::minMaxLoc(phase_diff_ideal23, &min, &max);

	wavelength23 = phase_diff(in[1], in[2], wavelengths[1], wavelengths[2], phase_diff_result23);
	phase_diff_ideal23 = (phase_diff_ideal23 - min) / (max - min + 0.2) * CV_2PI;
	phase_diff_result23 = (phase_diff_result23 - min) / (max - min + 0.2) * CV_2PI;

	// 123
	phase_diff(phase_diff_ideal12, phase_diff_ideal23, wavelength12, wavelength23, phase_diff_ideal123);
	cv::minMaxLoc(phase_diff_ideal123, &min, &max);

	phase_diff(phase_diff_result12, phase_diff_result23, wavelength12, wavelength23, out);
	phase_diff_ideal123 = (phase_diff_ideal123 - min) / (max - min + 0.2) * CV_2PI;
	out = (out - min) / (max - min + 0.02) * CV_2PI;
}


void DepthReconstructor::phase_reconstruct(const std::vector<cv::Mat>& in, cv::Mat& out)
{
	if (in.size() != this->strip_generator->phase_shift_number * this->strip_generator->wavelengths.size())
	{
		out = cv::Mat();
		return;
	}

	// generate ideal stripe to calculate min max of phase diff image
	std::vector<cv::Mat> strip_ideals;

	this->strip_generator->reset_index();
	for (int i = 0; i < this->strip_generator->phase_shift_number * this->strip_generator->wavelengths.size(); i++)
	{
		cv::Mat pattern;
		this->strip_generator->next(pattern);

		pattern.convertTo(pattern, CV_64FC1, 1.0);
		strip_ideals.push_back(pattern);
	}

	std::vector<cv::Mat> phase_results;
	std::vector<cv::Mat> phase_ideals;
	cv::Mat B_mask = cv::Mat(in[0].size(), CV_8UC1, 255);

	// phase shift reconstruct
	auto t0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < this->strip_generator->wavelengths.size(); i++)
	{
		std::vector<cv::Mat> image_for_same_wavelength(in.begin() + i * this->strip_generator->phase_shift_number, in.begin() + (i + 1) * this->strip_generator->phase_shift_number);
		std::vector<cv::Mat> ideal_for_same_wavelength(strip_ideals.begin() + i * this->strip_generator->phase_shift_number, strip_ideals.begin() + (i + 1) * this->strip_generator->phase_shift_number);

		cv::Mat phase_result, phase_ideal;
		phase_reconstruct_from_shift(image_for_same_wavelength, phase_result, B_mask, this->min_B);
		phase_reconstruct_from_shift(ideal_for_same_wavelength, phase_ideal, B_mask);

		//phase_result += CV_PI;

		phase_results.push_back(phase_result);
		phase_ideals.push_back(phase_ideal);
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	std::cout << "\ttime used for phase calc 1: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;

	// multi_wavelength merge
	t0 = std::chrono::high_resolution_clock::now();
	merge_multi_wavelength(phase_results, phase_ideals, this->strip_generator->wavelengths, out);
	t1 = std::chrono::high_resolution_clock::now();
	std::cout << "\ttime used for phase calc 2: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;

	// nomalize
	out.convertTo(out, CV_64FC1, 1 / CV_2PI);

	// mask
	out.setTo(0, ~B_mask);
}

void rectify(const cv::Mat& left, const cv::Mat& right, cv::Mat& left_rectified, cv::Mat& right_rectified, StereoCommon::StereoParameter* stereo_param)
{
	cv::Mat Kl = stereo_param->Kl;
	cv::Mat Dl = stereo_param->Dl;
	cv::Mat Kr = stereo_param->Kr;
	cv::Mat Dr = stereo_param->Dr;
	cv::Mat R = stereo_param->R;
	cv::Mat T = stereo_param->T;

	cv::Mat R1, R2, P1, P2, Q;
	cv::Size new_size = left.size();
	cv::stereoRectify(Kl, Dl, Kr, Dr, left.size(), R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 1, new_size);
	stereo_param->Pl = P1;
	stereo_param->Rl = R1;
	stereo_param->Pr = P2;
	stereo_param->Rr = R2;
	stereo_param->Q = Q;

	cv::Mat left_mapx, left_mapy, right_mapx, right_mapy;
	cv::initUndistortRectifyMap(Kl, Dl, R1, P1, new_size, CV_32FC1, left_mapx, left_mapy);
	cv::initUndistortRectifyMap(Kr, Dr, R2, P2, new_size, CV_32FC1, right_mapx, right_mapy);

	cv::remap(left, left_rectified, left_mapx, left_mapy, cv::INTER_LINEAR);
	cv::remap(right, right_rectified, right_mapx, right_mapy, cv::INTER_LINEAR);
}


double calcualte_ssd(const cv::Mat& left, const cv::Mat& right)
{
	cv::Mat diff = left - right;
	cv::Mat sqr_diff = diff.mul(diff);

	return sqrt(cv::sum(sqr_diff)[0]);
}


ushort stereo_search(const cv::Mat& left, const cv::Mat& right, const cv::Rect& left_block, int min, int max)
{
	if (left_block.width != left_block.height)
	{
		return 0;
	}

	const double phase_diff_threshold = 0.01;

	int ksize = left_block.width;
	int left_x = left_block.x + left_block.width / 2;
	int left_y = left_block.y + left_block.height / 2;
	double left_val = left.at<double>(left_y, left_x);

	double ssd_min = DBL_MAX;
	int ssd_min_idx = 0;
	for (int d = min; d < max; d++)
	{
#if 0
		if (left_x - d - ksize / 2 >= 0 && abs(left_val - right.at<double>(left_y, left_x - d)) < phase_diff_threshold)
		{
			double ssd = calcualte_ssd(left(left_block), right(cv::Rect(left_block.x - d, left_block.y, left_block.width, left_block.height)));
			if (ssd_min > ssd)
			{
				ssd_min = ssd;
				ssd_min_idx = d;
			}
		}
#else
		if (left_x - d - ksize / 2 >= 0)
		{
			double ssd = abs(left_val - right.at<double>(left_y, left_x - d));
			if (ssd_min > ssd)
			{
				ssd_min = ssd;
				ssd_min_idx = d;
			}
		}
#endif
	}

	return ssd_min_idx;
}


void match(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity, int min_disparity, int max_disparity)
{
	int w = left.cols;
	int h = left.rows;

	const int ksize = 3;

	disparity = cv::Mat::zeros(left.size(), CV_16UC1);

	for (int y = ksize / 2; y < h - ksize / 2; y++)
	{
		for (int x = ksize / 2; x < w - ksize / 2; x++)
		{
			double left_val = left.at<double>(y, x);
			if (left_val < DBL_EPSILON)
			{
				disparity.at<ushort>(y, x) = 0;
				continue;
			}

			cv::Rect block_left = cv::Rect(x - ksize / 2, y - ksize / 2, ksize, ksize);
			disparity.at<ushort>(y, x) = stereo_search(left, right, block_left, min_disparity, max_disparity);
		}
	}
}


void refine(const cv::Mat& left, const cv::Mat& right, const cv::Mat& disparity_ushort, cv::Mat& disparity_float, int min_disparity, int max_disparity)
{
	int w = left.cols;
	int h = left.rows;

	disparity_float = cv::Mat::zeros(disparity_ushort.size(), CV_64FC1);

	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			ushort d = disparity_ushort.at<ushort>(y, x);
#if 0
			// use 3 points to fit quafric curve
			if (d != 0 && d != min_disparity && d != max_disparity - 1 && (right.at<double>(y, x - d - 1) != 0) && (right.at<double>(y, x - d + 1) != 0))
			{
				double c0 = abs(left.at<double>(y, x) - right.at<double>(y, x - d));
				double c1 = abs(left.at<double>(y, x) - right.at<double>(y, x - (d - 1)));
				double c2 = abs(left.at<double>(y, x) - right.at<double>(y, x - (d + 1)));

				if (c1 >= c0 && c2 >= c0)
				{
					double demon = c1 + c2 - 2 * c0;
					double dsub = d + (c1 - c2) / demon / 2.0;

					disparity_float.at<double>(y, x) = dsub;
				}
				else
				{
					disparity_float.at<double>(y, x) = 0;
				}
			}
			else
			{
				disparity_float.at<double>(y, x) = 0;
			}
#else
			// use 5 points to fit linear curve
			if (d != 0 &&
				d > min_disparity + 1 &&
				d < max_disparity - 2 &&
				right.at<double>(y, x - d - 1) != 0 &&
				right.at<double>(y, x - d - 2) != 0 &&
				right.at<double>(y, x - d + 1) != 0 &&
				right.at<double>(y, x - d + 2) != 0)
			{
#if 0
				cv::Mat A(5, 2, CV_64FC1);
				cv::Mat B(5, 1, CV_64FC1);
				for (int i = -2; i <= 2; i++)
				{
					A.at<double>(i + 2, 0) = i;
					A.at<double>(i + 2, 1) = 1;

					B.at<double>(i + 2, 0) = (left.at<double>(y, x) - right.at<double>(y, x - (d + i)));
				}

				cv::Mat At = A.t();
				cv::Mat X = (At * A).inv() * (At * B);

				double a = X.at<double>(0);
				double b = X.at<double>(1);

				double dsub = d - b / a;
#else
				double c0 = (left.at<double>(y, x) - right.at<double>(y, x - d));
				double c1 = (left.at<double>(y, x) - right.at<double>(y, x - (d - 1)));
				double c2 = (left.at<double>(y, x) - right.at<double>(y, x - (d + 1)));

				double a = (c2 - c1) / 2;
				double b = c0;

				double dsub = d - b / a;
#endif
				if (abs(dsub - d) < 1)
				{
					disparity_float.at<double>(y, x) = dsub;
				}
				else
				{
					disparity_float.at<double>(y, x) = d;
				}
			}
			else
			{
				disparity_float.at<double>(y, x) = 0;
			}
#endif
		}
	}
}


void DepthReconstructor::depth_reconstruct(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity)
{
	const int min_disparity = 384;
	const int max_disparity = 512;

	// rectify
	cv::Mat left_rectified, right_rectified;
	auto t0 = std::chrono::high_resolution_clock::now();
	rectify(left, right, left_rectified, right_rectified, this->stereo_param);
	auto t1 = std::chrono::high_resolution_clock::now();
	std::cout << "\ttime used for rectify: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;

	// match
	cv::Mat disparity_ushort;
	t0 = std::chrono::high_resolution_clock::now();
	match(left_rectified, right_rectified, disparity_ushort, min_disparity, max_disparity);
	t1 = std::chrono::high_resolution_clock::now();
	std::cout << "\ttime used for match: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;

	// refine
	t0 = std::chrono::high_resolution_clock::now();
	refine(left_rectified, right_rectified, disparity_ushort, disparity, min_disparity, max_disparity);
	t1 = std::chrono::high_resolution_clock::now();
	std::cout << "\ttime used for refine: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;
	//disparity_ushort.convertTo(disparity, CV_64FC1, 1.0);
}
