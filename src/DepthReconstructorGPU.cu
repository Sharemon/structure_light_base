/**
 * Copyright @2023 Sharemon. All rights reserved.
 *
 @author: sharemon
 @date: 2023-10-24
 */

#include "DepthReconstructorGPU.hpp"
#include <cuda_runtime.h>

#define USE_GPU
#define GPU_BLOCK_SIZE (128)

/// @brief cuda api返回检查
#define CUDA_CHECK(call)                                                     \
    {                                                                        \
        const cudaError_t error = call;                                      \
        if (error != cudaSuccess)                                            \
        {                                                                    \
            printf("ERROR: %s:%d,", __FILE__, __LINE__);                     \
            printf("code:%d,reason:%s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                         \
        }                                                                    \
    }


using namespace StructureLightBase;


DepthReconstructorGPU::DepthReconstructorGPU(StripeGenerator* strip_generator, cv::Size image_size)
{
	this->strip_generator = strip_generator;
	this->min_B = 0;
	this->image_size = image_size;
	this->image_data_size = image_size.width * image_size.height;

	assert(this->strip_generator->wavelengths.size() == NUM_OF_WAVELENGTH);
	assert(this->strip_generator->phase_shift_number == NUM_OF_PHASE_SHIFT);

	// Initialize memory
#ifdef USE_GPU
	CUDA_CHECK(cudaMalloc((void **)&this->image_in, NUM_OF_PHASE_SHIFT * this->image_data_size * sizeof(float_custom_t)));

	CUDA_CHECK(cudaMalloc((void **)&this->phase_result_foreach_wavelength, NUM_OF_WAVELENGTH * this->image_data_size * sizeof(float_custom_t)));
	CUDA_CHECK(cudaMalloc((void **)&this->phase_diff_result, NUM_OF_WAVELENGTH * this->image_data_size * sizeof(float_custom_t)));

	this->phase_diff_result_host = (float_custom_t *)malloc(NUM_OF_WAVELENGTH * this->image_data_size * sizeof(float_custom_t));

	CUDA_CHECK(cudaMalloc((void**)&this->phase_result, this->image_data_size * sizeof(float_custom_t)));
	CUDA_CHECK(cudaMalloc((void**)&this->B_mask, this->image_data_size * sizeof(uchar)));
#else
	this->image_in = (float_custom_t*)malloc(NUM_OF_PHASE_SHIFT * this->image_data_size * sizeof(float_custom_t));

	this->phase_result_foreach_wavelength = (float_custom_t *)malloc(NUM_OF_WAVELENGTH * this->image_data_size * sizeof(float_custom_t));
	this->phase_diff_result = (float_custom_t *)malloc(NUM_OF_WAVELENGTH * this->image_data_size * sizeof(float_custom_t));

	this->phase_diff_result_host = (float_custom_t *)malloc(NUM_OF_WAVELENGTH * this->image_data_size * sizeof(float_custom_t));

	this->phase_result = (float_custom_t *)malloc(this->image_data_size * sizeof(float_custom_t));
	this->B_mask = (uchar *)malloc(this->image_data_size * sizeof(uchar));
#endif

	// calulate T
	this->T1 = this->strip_generator->wavelengths[0];
	this->T2 = this->strip_generator->wavelengths[1];
	this->T3 = this->strip_generator->wavelengths[2];
	this->T12 = (this->T1 * this->T2) / (this->T1 - this->T2);
	this->T23 = (this->T2 * this->T3) / (this->T2 - this->T3);
	this->T123 = (this->T12 * this->T23) / (this->T12 - this->T23);

	// calculate ideal phase max and min
	auto t0 = std::chrono::high_resolution_clock::now();
	this->calculate_ideal_phase_max_and_min();
	auto t1 = std::chrono::high_resolution_clock::now();
	std::cout << "\ttime used for calculate_ideal_phase_max_and_min: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;
}

DepthReconstructorGPU::~DepthReconstructorGPU()
{
	// Free memory
#ifdef USE_GPU
	cudaFree(this->image_in);

	cudaFree(this->phase_result_foreach_wavelength);
	cudaFree(this->phase_diff_result);

	free(this->phase_diff_result_host);
	
	cudaFree(this->phase_result);
	cudaFree(this->B_mask);
#else
	free(this->image_in);

	free(this->phase_result_foreach_wavelength);
	free(this->phase_diff_result);

	free(this->phase_diff_result_host);
	
	free(this->phase_result);
	free(this->B_mask);
#endif
}


void DepthReconstructorGPU::set_min_B(int B)
{
	this->min_B = B;
}


void DepthReconstructorGPU::set_stereo_param(StereoCommon::StereoParameter* parameter)
{
	this->stereo_param = parameter;
}

__global__ void phase_reconstruct_from_shift_gpu(const float_custom_t *in, float_custom_t *phase_result, uchar *B_mask, int size, int min_B)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float_custom_t phase_shift = (float_custom_t)CV_2PI / NUM_OF_PHASE_SHIFT;

	float_custom_t sum_sin = 0;
	float_custom_t sum_cos = 0;

	for (int j = 0; j < NUM_OF_PHASE_SHIFT; j++)
	{
		sum_sin += (float_custom_t)(in[j * size + i] * sin(j * phase_shift));
		sum_cos += (float_custom_t)(in[j * size + i] * cos(j * phase_shift));
	}

	float_custom_t B = sqrt(sum_sin * sum_sin + sum_cos * sum_cos) * 2 / NUM_OF_PHASE_SHIFT;
	if (B > min_B)
	{
		phase_result[i] = -atan2(sum_sin, sum_cos);
	}
	else
	{
		phase_result[i] = -CV_PI;
		B_mask[i] = 0;
	}
}		


void phase_reconstruct_from_shift(const float_custom_t *in, float_custom_t *phase_result, uchar *B_mask, int size, int min_B)
{
#ifdef USE_GPU
	phase_reconstruct_from_shift_gpu<<<size / GPU_BLOCK_SIZE, GPU_BLOCK_SIZE>>>(in, phase_result, B_mask, size, min_B);
	cudaDeviceSynchronize();
#else
	float_custom_t phase_shift = (float_custom_t)CV_2PI / NUM_OF_PHASE_SHIFT;

	for (int i = 0; i < size; i++)
	{
		float_custom_t sum_sin = 0;
		float_custom_t sum_cos = 0;

		for (int j = 0; j < NUM_OF_PHASE_SHIFT; j++)
		{
			sum_sin += (float_custom_t)(in[j * size + i] * sin(j * phase_shift));
			sum_cos += (float_custom_t)(in[j * size + i] * cos(j * phase_shift));
		}

		float_custom_t B = sqrt(sum_sin * sum_sin + sum_cos * sum_cos) * 2 / NUM_OF_PHASE_SHIFT;
		if (B > min_B)
		{
			phase_result[i] = -atan2(sum_sin, sum_cos);
		}
		else
		{
			phase_result[i] = -CV_PI;
			B_mask[i] = 0;
		}
	}
#endif
}

__global__ void phase_diff_gpu(const float_custom_t *in1, const float_custom_t *in2, float_custom_t *out, float_custom_t T1, float_custom_t T2)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float_custom_t t = T2 / (T2 - T1);

	float_custom_t phi1 = in1[i];
	float_custom_t phi2 = in2[i];

	float_custom_t delta_phi = phi1 > phi2 ? phi1 - phi2 : (float_custom_t)CV_2PI - (phi2 - phi1);
	int k = (int)round((t * delta_phi - phi1) / (float_custom_t)CV_2PI);

	out[i] = k * CV_2PI + phi1;
}


void phase_diff(const float_custom_t *in1, const float_custom_t *in2, float_custom_t *out, int size, float_custom_t T1, float_custom_t T2)
{
#ifdef USE_GPU
	phase_diff_gpu<<<size / GPU_BLOCK_SIZE, GPU_BLOCK_SIZE>>>(in1, in2, out, T1, T2);
	cudaDeviceSynchronize();
#else
	float_custom_t t = T2 / (T2 - T1);

	for (int i = 0; i < size; i++)
	{
		float_custom_t phi1 = in1[i];
		float_custom_t phi2 = in2[i];

		float_custom_t delta_phi = phi1 > phi2 ? phi1 - phi2 : (float_custom_t)CV_2PI - (phi2 - phi1);
		int k = (int)round((t * delta_phi - phi1) / (float_custom_t)CV_2PI);

		out[i] = k * CV_2PI + phi1;
	}

#endif
}


void find_max_and_min(const float_custom_t *data, float_custom_t *max, float_custom_t *min, int size)
{
	float_custom_t max_local = data[0];
	float_custom_t min_local = data[0];

	for (int i = 1; i < size; i++)
	{
		max_local = data[i] > max_local ? data[i] : max_local;
		min_local = data[i] < min_local ? data[i] : min_local;
	}

	*max = max_local; 
	*min = min_local;
}

__global__ void phase_normalize_gpu(float_custom_t *data, float_custom_t max, float_custom_t min, float_custom_t scale)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	float_custom_t range_inv = 1.0 / (max - min + 0.2); 	// add 0.2 to avoid max == min

	data[i] = (data[i] - min) * range_inv * scale;
}


void phase_normalize(float_custom_t *data, float_custom_t max, float_custom_t min, int size, float_custom_t scale = 1)
{
#ifdef USE_GPU
	phase_normalize_gpu<<<size / GPU_BLOCK_SIZE, GPU_BLOCK_SIZE>>>(data, max, min, scale);
	cudaDeviceSynchronize();
#else
	float_custom_t range_inv = 1.0 / (max - min + 0.2); 	// add 0.2 to avoid max == min

	for (int i = 0; i < size; i++)
	{
		data[i] = (data[i] - min) * range_inv * scale;
	}
#endif
}

__global__ void B_filter_gpu(float_custom_t *in, uchar *mask, float_custom_t *out)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	out[i] = mask[i] == 0 ? 0 : in[i];
}


void B_filter(float_custom_t *in, uchar *mask, float_custom_t *out, int size)
{
#ifdef USE_GPU
	B_filter_gpu<<<size / GPU_BLOCK_SIZE, GPU_BLOCK_SIZE>>>(in, mask, out);
	cudaDeviceSynchronize();
#else
	for (int i = 0; i < size; i++)
	{
		out[i] = mask[i] == 0 ? 0 : in[i];
	}
#endif
}


void DepthReconstructorGPU::calculate_ideal_phase_max_and_min()
{
	// generate ideal stripe to calculate min max of phase diff image
	std::vector<cv::Mat> strip_ideals;

	this->strip_generator->reset_index();
	for (int i = 0; i < NUM_OF_PHASE_SHIFT * NUM_OF_WAVELENGTH; i++)
	{
		cv::Mat pattern;
		this->strip_generator->next(pattern);

		pattern.convertTo(pattern, CV_FLOAT_CUSTOM, 1.0);
		strip_ideals.push_back(pattern);
	}
	
	// phase shift reconstruct
	for (int i = 0; i < NUM_OF_WAVELENGTH; i++)
	{
		for (int j = 0; j < NUM_OF_PHASE_SHIFT; j++)
		{
#ifdef USE_GPU
			cudaMemcpy((void*)(this->image_in + j * this->image_data_size), (void*)strip_ideals[i * NUM_OF_PHASE_SHIFT + j].data, this->image_data_size * sizeof(float_custom_t), cudaMemcpyHostToDevice);
#else
			memcpy((void*)(this->image_in + j * this->image_data_size), (void*)strip_ideals[i * NUM_OF_PHASE_SHIFT + j].data, this->image_data_size * sizeof(float_custom_t));
#endif
		}

		phase_reconstruct_from_shift(	this->image_in, this->phase_result_foreach_wavelength + i * this->image_data_size, 
										this->B_mask, this->image_data_size, -1);
	}

	// phase diff
	phase_diff(	this->phase_result_foreach_wavelength + 0 * this->image_data_size,
				this->phase_result_foreach_wavelength + 1 * this->image_data_size,
				this->phase_diff_result, 
				this->image_data_size, this->T1, this->T2);

#ifdef USE_GPU
	cudaMemcpy((void *)this->phase_diff_result_host, (void *)this->phase_diff_result, this->image_data_size * sizeof(float_custom_t), cudaMemcpyDeviceToHost);
#else
	memcpy((void *)this->phase_diff_result_host, (void *)this->phase_diff_result, this->image_data_size * sizeof(float_custom_t));
#endif
	find_max_and_min(this->phase_diff_result_host, &this->max_T12, &this->min_T12, this->image_data_size);

	phase_normalize(this->phase_diff_result, this->max_T12, this->min_T12, this->image_data_size, CV_2PI);

	phase_diff(	this->phase_result_foreach_wavelength + 1 * this->image_data_size,
				this->phase_result_foreach_wavelength + 2 * this->image_data_size,
				this->phase_diff_result + this->image_data_size, 
				this->image_data_size, this->T2, this->T3);

#ifdef USE_GPU
	cudaMemcpy((void *)(this->phase_diff_result_host + this->image_data_size), (void *)(this->phase_diff_result + this->image_data_size), this->image_data_size * sizeof(float_custom_t), cudaMemcpyDeviceToHost);
#else
	memcpy((void *)(this->phase_diff_result_host + this->image_data_size), (void *)(this->phase_diff_result + this->image_data_size), this->image_data_size * sizeof(float_custom_t));
#endif
	find_max_and_min(this->phase_diff_result_host + this->image_data_size, &this->max_T23, &this->min_T23, this->image_data_size);

	phase_normalize(this->phase_diff_result + this->image_data_size, this->max_T23, this->min_T23, this->image_data_size, CV_2PI);

	phase_diff( this->phase_diff_result, 
				this->phase_diff_result + this->image_data_size,
				this->phase_diff_result + this->image_data_size * 2,
				this->image_data_size, this->T12, this->T23);

#ifdef USE_GPU
	cudaMemcpy((void *)(this->phase_diff_result_host + this->image_data_size * 2), (void *)(this->phase_diff_result + this->image_data_size * 2), this->image_data_size * sizeof(float_custom_t), cudaMemcpyDeviceToHost);
#else
	memcpy((void *)(this->phase_diff_result_host + this->image_data_size * 2), (void *)(this->phase_diff_result + this->image_data_size * 2), this->image_data_size * sizeof(float_custom_t));
#endif
	find_max_and_min(this->phase_diff_result_host + this->image_data_size * 2, &this->max_T123, &this->min_T123, this->image_data_size);

	std::cout << "max: " << this->max_T12 << ", " << this->max_T23 << ", " << this->max_T123 << std::endl;
	std::cout << "min: " << this->min_T12 << ", " << this->min_T23 << ", " << this->min_T123 << std::endl;	
}


void DepthReconstructorGPU::phase_reconstruct(const std::vector<cv::Mat>& in, cv::Mat& out)
{
	if (in.size() != this->strip_generator->phase_shift_number * this->strip_generator->wavelengths.size())
	{
		out = cv::Mat();
		return;
	}

#ifdef USE_GPU
	cudaMemset(this->B_mask, 255, this->image_data_size * sizeof(uchar));
#else
	memset(this->B_mask, 255, this->image_data_size * sizeof(uchar));
#endif

	// phase shift reconstruct
	for (int i = 0; i < NUM_OF_WAVELENGTH; i++)
	{
		auto t0 = std::chrono::high_resolution_clock::now();
		for (int j = 0; j < NUM_OF_PHASE_SHIFT; j++)
		{
#ifdef USE_GPU
			cudaMemcpy((void*)(this->image_in + j * this->image_data_size), (void*)in[i * NUM_OF_PHASE_SHIFT + j].data, this->image_data_size * sizeof(float_custom_t), cudaMemcpyHostToDevice);
#else
			memcpy((void*)(this->image_in + j * this->image_data_size), (void*)in[i * NUM_OF_PHASE_SHIFT + j].data, this->image_data_size * sizeof(float_custom_t));
#endif
		}
		auto t1 = std::chrono::high_resolution_clock::now();
		std::cout << "\ttime used for cudaMemcpy in: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;

		t0 = std::chrono::high_resolution_clock::now();
		phase_reconstruct_from_shift(this->image_in, this->phase_result_foreach_wavelength + i * this->image_data_size,
									 this->B_mask, this->image_data_size, this->min_B);
		t1 = std::chrono::high_resolution_clock::now();
		std::cout << "\ttime used for phase_reconstruct_from_shift: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;
	}

	// phase diff
	auto t0 = std::chrono::high_resolution_clock::now();
	phase_diff(this->phase_result_foreach_wavelength + 0 * this->image_data_size,
			   this->phase_result_foreach_wavelength + 1 * this->image_data_size,
			   this->phase_diff_result,
			   this->image_data_size, this->T1, this->T2);
	auto t1 = std::chrono::high_resolution_clock::now();
	std::cout << "\ttime used for phase_diff: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;

	t0 = std::chrono::high_resolution_clock::now();
	phase_normalize(this->phase_diff_result, this->max_T12, this->min_T12, this->image_data_size, CV_2PI);
	t1 = std::chrono::high_resolution_clock::now();
	std::cout << "\ttime used for phase_normalize: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;

	t0 = std::chrono::high_resolution_clock::now();
	phase_diff(	this->phase_result_foreach_wavelength + 1 * this->image_data_size,
				this->phase_result_foreach_wavelength + 2 * this->image_data_size,
				this->phase_diff_result + this->image_data_size,
				this->image_data_size, this->T2, this->T3);
	t1 = std::chrono::high_resolution_clock::now();
	std::cout << "\ttime used for phase_diff: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;

	t0 = std::chrono::high_resolution_clock::now();
	phase_normalize(this->phase_diff_result + this->image_data_size, this->max_T23, this->min_T23, this->image_data_size, CV_2PI);
	t1 = std::chrono::high_resolution_clock::now();
	std::cout << "\ttime used for phase_normalize: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;

	t0 = std::chrono::high_resolution_clock::now();
	phase_diff( this->phase_diff_result,
				this->phase_diff_result + this->image_data_size,
				this->phase_diff_result + this->image_data_size * 2,
				this->image_data_size, this->T12, this->T23);
	t1 = std::chrono::high_resolution_clock::now();
	std::cout << "\ttime used for phase_diff: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;

	t0 = std::chrono::high_resolution_clock::now();
	phase_normalize(this->phase_diff_result + this->image_data_size * 2, this->max_T123, this->min_T123, this->image_data_size, CV_2PI);
	t1 = std::chrono::high_resolution_clock::now();
	std::cout << "\ttime used for phase_normalize: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;

	// filter by modulation amplitude
	t0 = std::chrono::high_resolution_clock::now();
	B_filter(this->phase_diff_result + this->image_data_size * 2, this->B_mask, this->phase_result, this->image_data_size);
	t1 = std::chrono::high_resolution_clock::now();
	std::cout << "\ttime used for B_filter: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;

	// copy result to out
	t0 = std::chrono::high_resolution_clock::now();
	out = cv::Mat::zeros(this->image_size, CV_FLOAT_CUSTOM);
#ifdef USE_GPU
	cudaMemcpy(out.data, this->phase_result, this->image_data_size * sizeof(float_custom_t), cudaMemcpyDeviceToHost);
#else
	memcpy(out.data, this->phase_result, this->image_data_size * sizeof(float_custom_t));
#endif
	t1 = std::chrono::high_resolution_clock::now();
	std::cout << "\ttime used for cudaMemcpy out: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "ms" << std::endl;
}


// reuse cpu version stereo match as it only cost ~80ms
extern void rectify(const cv::Mat& left, const cv::Mat& right, cv::Mat& left_rectified, cv::Mat& right_rectified, StereoCommon::StereoParameter* stereo_param);
extern double calcualte_ssd(const cv::Mat& left, const cv::Mat& right);
extern ushort stereo_search(const cv::Mat& left, const cv::Mat& right, const cv::Rect& left_block, int min, int max);
extern void match(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity, int min_disparity, int max_disparity);
extern void refine(const cv::Mat& left, const cv::Mat& right, const cv::Mat& disparity_ushort, cv::Mat& disparity_float, int min_disparity, int max_disparity);

void DepthReconstructorGPU::depth_reconstruct(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity)
{
	 const int min_disparity = 384;
	 const int max_disparity = 512;

	 // rectify
	 cv::Mat left_rectified, right_rectified;
	 rectify(left, right, left_rectified, right_rectified, this->stereo_param);

	 // match
	 cv::Mat disparity_ushort;
	 match(left_rectified, right_rectified, disparity_ushort, min_disparity, max_disparity);

	 // refine
	 refine(left_rectified, right_rectified, disparity_ushort, disparity, min_disparity, max_disparity);
	 //disparity_ushort.convertTo(disparity, CV_FLOAT_CUSTOM, 1.0);
}
