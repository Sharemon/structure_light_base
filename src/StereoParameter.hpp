/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-10-27
 */

#pragma once

#include <string>
#include <opencv2/opencv.hpp>

namespace StereoCommon
{
    class StereoParameter
    {
    public:
        cv::Mat Kl;
        cv::Mat Dl;
        cv::Mat Kr;
        cv::Mat Dr;
        cv::Mat R;
        cv::Mat T;
        cv::Mat Rl;
        cv::Mat Pl;
        cv::Mat Rr;
        cv::Mat Pr;
        cv::Mat Q;
        cv::Size image_size;

        StereoParameter(std::string filename, cv::Size image_size)
        {
            this->image_size = image_size;

            cv::FileStorage param_file(filename, cv::FileStorage::READ);

            param_file["Kl"] >> this->Kl;
            param_file["Dl"] >> this->Dl;
            param_file["Kr"] >> this->Kr;
            param_file["Dr"] >> this->Dr;
            param_file["R"] >> this->R;
            param_file["T"] >> this->T;
        }
        ~StereoParameter() {}
    };
   
}