/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-12-04
 */

#pragma once

#include "StripeGenerator.hpp"
#include "StereoParameter.hpp"

#define NUM_OF_WAVELENGTH (3)
#define NUM_OF_PHASE_SHIFT (12)

namespace StructureLightBase
{
    class DepthReconstructor
    {
    private:
        StripeGenerator* strip_generator = NULL;
        int min_B;
        StereoCommon::StereoParameter* stereo_param;

    public:
        DepthReconstructor(StripeGenerator *strip_generator);
        ~DepthReconstructor();
        
        void set_min_B(int B);
        void set_stereo_param(StereoCommon::StereoParameter* parameter);
        void phase_reconstruct(const std::vector<cv::Mat>& in, cv::Mat& out);
        void depth_reconstruct(const cv::Mat& left, const cv::Mat& right, cv::Mat& disparity);
    };
}
