/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-10-24
 */

#ifndef __DEPTH_RECONSTRUCTOR_HPP__
#define __DEPTH_RECONSTRUCTOR_HPP__

#include "StripeGenerator.hpp"
#include "StereoParameter.hpp"

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

#endif