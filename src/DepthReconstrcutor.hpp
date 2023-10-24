/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-10-24
 */

#ifndef __DEPTH_RECONSTRUCTOR_HPP__
#define __DEPTH_RECONSTRUCTOR_HPP__

#include "StripeGenerator.hpp"

namespace StructureLightBase
{
    class DepthReconstructor
    {
    private:
        StripeGenerator* strip_generator = NULL;
        int min_B;

    public:
        DepthReconstructor(StripeGenerator *strip_generator);
        ~DepthReconstructor();
        
        void set_min_B(int B);
        void reconstruct(const std::vector<cv::Mat>& in, cv::Mat& out);
    };
}

#endif