/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-10-24
 */

#ifndef __STRIPE_GENERATOR_HPP__
#define __STRIPE_GENERATOR_HPP__

#include <vector>
#include <opencv2/opencv.hpp>

namespace StructureLightBase
{
    enum StripeType {
        MULTI_WAVELENGTH_HETERODYNE,
        GRAY_CODE,
        COMPLEMENTARY_GRAY_CODE
    };

    class StripeGenerator
    {
    private:
        int width;
        int height;
        StripeType type;
        std::vector<int> wavelengths;
        int phase_shift_number;
        int A;
        int B;
        int current_wavelength_idx;
        int current_phase_shift_idx;

        void reset_index();

    public:
        StripeGenerator(int width, int height);
        ~StripeGenerator();

        void set_type(StripeType type);
        void set_waveLengths(const std::vector<int>& wavelengths);
        void set_phase_shift_number(int N);
        void set_A(int A);
        void set_B(int B);

        bool next(cv::Mat& pattern);
        int get_pattern_size();

        friend class DepthReconstructor;
    };
    
}

#endif