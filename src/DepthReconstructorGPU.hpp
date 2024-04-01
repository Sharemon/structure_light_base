/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-12-04
 */

#pragma once

#include <fstream>
#include "StripeGenerator.hpp"
#include "StereoParameter.hpp"

#define NUM_OF_WAVELENGTH (3)
#define NUM_OF_PHASE_SHIFT (12)

#define CV_FLOAT_CUSTOM (CV_64FC1)
typedef double float_custom_t;

namespace StructureLightBase
{
    class DepthReconstructorGPU
    {
    private:
        StripeGenerator* strip_generator = NULL;
        int min_B;
        StereoCommon::StereoParameter* stereo_param;
        cv::Size image_size;
        int image_data_size;

        float_custom_t *image_in;                            // 12 * width * height * sizeof(uchar)
        float_custom_t *phase_result_foreach_wavelength;     //  3 * width * height * sizeof(float_custom_t)
        float_custom_t *phase_diff_result;                   //  3 * width * height * sizeof(float_custom_t)
        float_custom_t *phase_diff_result_host;              //  3 * width * height * sizeof(float_custom_t)
        float_custom_t *phase_result;                        //  1 * width * height * sizeof(float_custom_t)
        uchar *B_mask;                                       //  1 * width * height * sizeof(uchar)

        float_custom_t max_T12, max_T23, max_T123;
        float_custom_t min_T12, min_T23, min_T123;
        float_custom_t T1, T2, T3, T12, T23, T123;

        void calculate_ideal_phase_max_and_min();

    public:
        DepthReconstructorGPU(StripeGenerator *strip_generator, cv::Size image_size);
        ~DepthReconstructorGPU();
        
        void set_min_B(int B);
        void set_stereo_param(StereoCommon::StereoParameter* parameter);
        void phase_reconstruct(const std::vector<cv::Mat>& in, cv::Mat& out);
        void depth_reconstruct(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity);
    };
}
