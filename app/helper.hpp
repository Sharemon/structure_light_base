/**
 * Copyright @2023 Sharemon. All rights reserved.
 *
 @author: sharemon
 @date: 2023-10-31
 */

#pragma once

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "../src/StripeGenerator.hpp"
#ifdef __linux__
    #include <getopt.h>
#elif defined(_WIN32)
    #include "./getopt.h"
#endif

class Camera
{
private:
    cv::VideoCapture *cap;

public:
    Camera(int id, int width, int height, int exposure)
    {
        cap = new cv::VideoCapture(id);

        cap->set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap->set(cv::CAP_PROP_FRAME_HEIGHT, width);
        cap->set(cv::CAP_PROP_EXPOSURE, exposure);
    }
    ~Camera()
    {
        cap->release();
    }

    bool capture(cv::Mat &image)
    {
        return cap->read(image);
    }
};

class Projector
{
private:
    /* data */
public:
    Projector(/* args */)
    {
        cv::namedWindow("projector", cv::WINDOW_NORMAL);
    }
    ~Projector() {}

    void project(const cv::Mat &pattern)
    {
        cv::imshow("projector", pattern);
        cv::moveWindow("projector", 1920, 0);
        cv::setWindowProperty("projector", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        cv::waitKey(1);
    }
};

class Args
{
public:
    StructureLightBase::StripeType type;
    std::vector<int> wavelengths;
    int phase_shift_number;
    int gray_code_bits;
    int A;
    int B;
    int Bmin;
    std::string input_folder;

    Args() {}
    ~Args() {}
};

template <typename T>
void noise_filter(const cv::Mat &src, cv::Mat &dst, const T value_threshold = 0.05)
{
    int w = src.cols;
    int h = src.rows;

    const int ksize = 5;
    const int count_threshold = 2;

    dst = cv::Mat::zeros(src.size(), src.type());

    for (int y = ksize / 2; y < h - ksize / 2; y++)
    {
        for (int x = ksize / 2; x < w - ksize / 2; x++)
        {
            T val = src.at<T>(y, x);
            int valid_cnt = 0;
            for (int i = -ksize / 2; i <= ksize / 2; i++)
            {
                for (int j = ksize / 2; j <= ksize / 2; j++)
                {
                    if (abs(src.at<T>(y + i, x + j) - val) < value_threshold)
                    {
                        valid_cnt++;
                    }
                }
            }

            if (valid_cnt >= count_threshold)
            {
                dst.at<T>(y, x) = val;
            }
        }
    }
}

template <typename T>
void save_image_to_txt(std::string save_path, const cv::Mat &img)
{
    int w = img.cols;
    int h = img.rows;

    std::ofstream outf(save_path);
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            outf << img.at<T>(y, x) << " ";
        }

        outf << std::endl;
    }

    outf.close();
}

template <typename T>
void convert_disparity_map_to_point_cloud(const cv::Mat &disp, std::vector<cv::Vec6f> &point_cloud_with_texture, const cv::Mat &Q, const cv::Mat &texture)
{
    point_cloud_with_texture.clear();

    int width = disp.cols;
    int height = disp.rows;

    double cx = -Q.at<double>(0, 3);
    double cy = -Q.at<double>(1, 3);
    double f = Q.at<double>(2, 3);
    double w = Q.at<double>(3, 2);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float d = disp.at<T>(y, x);
            if (d < 20) // set a distance max limit of 3m
                continue;

            float dw = d * w;

            float X = (x - cx) / dw;
            float Y = (y - cy) / dw;
            float Z = f / dw;

            cv::Vec6f xyz_rgb;
            xyz_rgb[0] = X;
            xyz_rgb[1] = Y;
            xyz_rgb[2] = Z;

            if (texture.channels() == 1)
            {
                uchar grayscale = texture.at<uchar>(y, x);
                xyz_rgb[3] = grayscale;
                xyz_rgb[4] = grayscale;
                xyz_rgb[5] = grayscale;
            }
            else
            {
                cv::Vec3b rgb = texture.at<cv::Vec3b>(y, x);
                xyz_rgb[3] = rgb[0];
                xyz_rgb[4] = rgb[1];
                xyz_rgb[5] = rgb[2];
            }

            point_cloud_with_texture.push_back(xyz_rgb);
        }
    }
}

void parse_args(int argc, char *argv[], Args& args)
{
    int opt = 0;
    const char *opt_string = "";
    int opt_indx = 0;
    char* cstrtmp = NULL;
    static struct option long_options[] = {
        {"type", required_argument, NULL, 't'},
        {"wavelngth", required_argument, NULL, 'w'},
        {"phase_shift_number", required_argument, NULL, 'p'},
        {"gray_code_bits", required_argument, NULL, 'g'},
        {"A", required_argument, NULL, 'a'},
        {"B", required_argument, NULL, 'b'},
        {"Bmin", required_argument, NULL, 'm'},
        {"inputfolder", required_argument, NULL, 'f'}};

    while ((opt = getopt_long_only(argc, argv, opt_string, long_options, &opt_indx)) != -1)
    {
        printf("opt is %c, arg is %s\n", opt, optarg);
        switch (opt)
        {
        case 't':
            if (std::string(optarg) == "multi-wavelength")
            {
                args.type = StructureLightBase::MULTI_WAVELENGTH_HETERODYNE;
            }
            else if (std::string(optarg) == "gray-code")
            {
                args.type = StructureLightBase::GRAY_CODE;
            }
            else if (std::string(optarg) == "complementary-gray-code")
            {
                args.type = StructureLightBase::COMPLEMENTARY_GRAY_CODE;
            }
            else
            {
                args.type = StructureLightBase::MULTI_WAVELENGTH_HETERODYNE;
            }
            break;
        case 'w':
            cstrtmp = strtok(optarg, ",");  
            while (cstrtmp != NULL) {
                args.wavelengths.push_back(atoi(cstrtmp));
                cstrtmp = strtok(NULL, ",");
            }
            break;
        case 'p':
            args.phase_shift_number = atoi(optarg);
            break;
        case 'g':
            args.gray_code_bits = atoi(optarg);
            break;
        case 'a':
            args.A = atoi(optarg);
            break;
        case 'b':
            args.B = atoi(optarg);
            break;
        case 'm':
            args.Bmin = atoi(optarg);
            break;
        case 'f':
            args.input_folder = std::string(optarg);
        default:
            printf("unknown option.\n");
            break;
        }
    }
}
