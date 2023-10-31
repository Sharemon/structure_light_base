/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-10-24
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "../src/DepthReconstrcutor.hpp"
#include "../src/StripeGenerator.hpp"
#include "../src/StereoParameter.hpp"


class Camera
{
private:
    cv::VideoCapture *cap;
public:
    Camera(int id, int width, int height, int exposure){
        cap = new cv::VideoCapture(id);

        cap->set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap->set(cv::CAP_PROP_FRAME_HEIGHT, width);
        cap->set(cv::CAP_PROP_EXPOSURE, exposure);
    }
    ~Camera() {
        cap->release();
    }

    bool capture(cv::Mat& image)
    {
        return cap->read(image);
    }
};

class Projector
{
private:
    /* data */
public:
    Projector(/* args */){
        cv::namedWindow("projector", cv::WINDOW_NORMAL);
    }
    ~Projector(){}

    void project(const cv::Mat& pattern) {
        cv::imshow("projector", pattern);
        cv::moveWindow("projector", 1920, 0);
        cv::setWindowProperty("projector", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        cv::waitKey(1);
    }
};

using namespace StructureLightBase;

template<typename T>
void noise_filter(const cv::Mat&src, cv::Mat& dst, const T value_threshold = 0.05)
{
    int w = src.cols;
    int h = src.rows;

    const int ksize = 5;
    const int count_threshold = 2;

    dst = cv::Mat::zeros(src.size(), src.type());

    for (int y = ksize / 2; y < h - ksize/2; y++)
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

template<typename T>
void save_image_to_txt(std::string save_path, const cv::Mat& img)
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


void test_multi_wavelength_heterodyne()
{
    // initialize
    Camera *cam = new Camera(0, 1280, 720, -13);
    Projector *projector = new Projector;

    std::vector<int> wavelengths = { 24, 26, 28 };
    StripeGenerator *stripe_generator = new StripeGenerator(1920, 1080);
    stripe_generator->set_type(MULTI_WAVELENGTH_HETERODYNE);     // GRAY_CODE
    stripe_generator->set_waveLengths(wavelengths);
    stripe_generator->set_phase_shift_number(15);
    stripe_generator->set_A(128);
    stripe_generator->set_B(60);

    // project stripe and capture image
    cv::Mat image;
    std::vector<cv::Mat> images;
    
    for (int i=0;i<stripe_generator->get_pattern_size();i++)
    {
        cv::Mat pattern;
        if (!stripe_generator->next(pattern))
        {
            std::cout << "run out of stripe." << std::endl;
            break;
        }

        projector->project(pattern);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        // capture one frame
        cam->capture(image);
        // capture second frame for reconstruction
        if (cam->capture(image))
        {
            cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);

            cv::imwrite("./" + std::to_string(i) + ".png", image);
            cv::imshow("phase unwrap", image);
            cv::waitKey(1);
            images.push_back(image);
        }
        else
        {
            std::cout << "cannot read " << i << "th image." << std::endl;
        }
    }

    // unwrap phase and reconstruct depth
    DepthReconstructor *depth_reconstructor = new DepthReconstructor(stripe_generator);
    depth_reconstructor->set_min_B(100);
    
    cv::Mat phase_image;
    depth_reconstructor->phase_reconstruct(images, phase_image);

    cv::Mat phase_image_filtered;
    noise_filter<double>(phase_image, phase_image_filtered);

    if (!phase_image.empty())
    {
        // save result
        save_image_to_txt<double>("./result.txt", phase_image_filtered);

        // show result
        cv::Mat phase_image_u8;
        phase_image_filtered.convertTo(phase_image_u8, CV_8UC1, 255);

        cv::imshow("phase unwrap", phase_image_u8);
        cv::waitKey(0);
    }
}


void test_multi_wavelength_heterodyne_from_file()
{
    // initialize
#if 0
    std::vector<int> wavelengths = { 28, 26, 24 };
    StripeGenerator* stripe_generator = new StripeGenerator(1280, 1024);
    stripe_generator->set_type(MULTI_WAVELENGTH_HETERODYNE);     // GRAY_CODE
    stripe_generator->set_waveLengths(wavelengths);
    stripe_generator->set_phase_shift_number(12);
    stripe_generator->set_A(130);
    stripe_generator->set_B(90);

    // project stripe and capture image
    std::vector<cv::Mat> images;

    for (int i = 0; i < stripe_generator->get_pattern_size(); i++)
    {
        std::string filepath = "../data/2/" + std::to_string(i + 1) + ".bmp";
        cv::Mat image = cv::imread(filepath, cv::IMREAD_GRAYSCALE);

        if (!image.empty())
        {
            images.push_back(image);
        }
    }
#else
    std::vector<int> wavelengths = { 24, 26, 28 };
    StripeGenerator* stripe_generator = new StripeGenerator(1920, 1080);
    stripe_generator->set_type(MULTI_WAVELENGTH_HETERODYNE);     // GRAY_CODE
    stripe_generator->set_waveLengths(wavelengths);
    stripe_generator->set_phase_shift_number(15);
    stripe_generator->set_A(128);
    stripe_generator->set_B(60);

    // project stripe and capture image
    std::vector<cv::Mat> images;

    for (int i = 0; i < stripe_generator->get_pattern_size(); i++)
    {
        std::string filepath = "../data/1/" + std::to_string(i) + ".png";
        cv::Mat image = cv::imread(filepath, cv::IMREAD_GRAYSCALE);

        if (!image.empty())
        {
            images.push_back(image);
        }
    }
#endif

    // unwrap phase and reconstruct depth
    DepthReconstructor* depth_reconstructor = new DepthReconstructor(stripe_generator);
    depth_reconstructor->set_min_B(10);

    cv::Mat phase_image;
    depth_reconstructor->phase_reconstruct(images, phase_image);

    cv::Mat phase_image_filtered;
    //noise_filter<double>(phase_image, phase_image_filtered);
    phase_image_filtered = phase_image;

    if (!phase_image.empty())
    {
        // save result
        save_image_to_txt<double>("./result.txt", phase_image_filtered);

        // show result
        cv::Mat phase_image_u8;
        phase_image_filtered.convertTo(phase_image_u8, CV_8UC1, 255);

        cv::imshow("phase unwrap", phase_image_u8);
        cv::waitKey(0);
    }
}


void test_stereo_multi_wavelength_heterodyne()
{
    // initialize
    Camera *cam = new Camera(0, 2560, 720, -13);
    Projector *projector = new Projector;

    std::vector<int> wavelengths = { 24, 26, 28 };
    StripeGenerator* stripe_generator = new StripeGenerator(1920, 1080);
    stripe_generator->set_type(MULTI_WAVELENGTH_HETERODYNE);     // GRAY_CODE
    stripe_generator->set_waveLengths(wavelengths);
    stripe_generator->set_phase_shift_number(15);
    stripe_generator->set_A(128);
    stripe_generator->set_B(60);

    // project stripe and capture image
    std::vector<cv::Mat> images_left;
    std::vector<cv::Mat> images_right;
    
    for (int i=0;i<stripe_generator->get_pattern_size();i++)
    {
        cv::Mat pattern;
        if (!stripe_generator->next(pattern))
        {
            std::cout << "run out of stripe." << std::endl;
            break;
        }

        projector->project(pattern);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        // capture one frame
        cv::Mat image;
        cam->capture(image);
        // capture second frame for reconstruction
        if (cam->capture(image))
        {
            cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);

            cv::Mat image_left = image.colRange(0, image.cols/2);
            cv::Mat image_right = image.colRange(image.cols/2, image.cols);

            images_left.push_back(image_left);
            images_right.push_back(image_right);
        }
        else
        {
            std::cout << "cannot read " << i << "th image." << std::endl;
        }
    }

    StereoCommon::StereoParameter* stereo_param = new StereoCommon::StereoParameter("../data/3/stereo_params.yaml", images_left[0].size());

    // unwrap phase
    DepthReconstructor* depth_reconstructor = new DepthReconstructor(stripe_generator);
    depth_reconstructor->set_min_B(10);

    cv::Mat phase_image_left;
    depth_reconstructor->phase_reconstruct(images_left, phase_image_left);

    cv::Mat phase_image_left_filtered;
    //noise_filter<double>(phase_image_left, phase_image_left_filtered);
    phase_image_left_filtered = phase_image_left;

    cv::Mat phase_image_right;
    depth_reconstructor->phase_reconstruct(images_right, phase_image_right);

    cv::Mat phase_image_right_filtered;
    //noise_filter<double>(phase_image_right, phase_image_right_filtered);
    phase_image_right_filtered = phase_image_right;

    // depth reconstruct
    cv::Mat disparity;
    depth_reconstructor->set_stereo_param(stereo_param);
    depth_reconstructor->depth_reconstruct(phase_image_left_filtered, phase_image_right_filtered, disparity);

    cv::Mat disparity_filtered;
    //noise_filter<double>(disparity, disparity_filtered, 5);
    disparity_filtered = disparity;

    if (!disparity_filtered.empty())
    {
        // save result
        save_image_to_txt<double>("./disparity.txt", disparity_filtered);

        // show result
        cv::Mat disparity_u8;
        disparity_filtered.convertTo(disparity_u8, CV_8UC1, 0.5);

        cv::imshow("disparity", disparity_u8);
        cv::waitKey(0);
    }
}


template<typename T>
void convert_disparity_map_to_point_cloud(const cv::Mat& disp, std::vector<cv::Vec6f>& point_cloud_with_texture, const cv::Mat& Q, const cv::Mat& texture)
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

void test_stereo_multi_wavelength_heterodyne_from_file()
{
    // initialize
    std::vector<int> wavelengths = { 28, 26, 24 };
    StripeGenerator* stripe_generator = new StripeGenerator(1280, 1024);
    stripe_generator->set_type(MULTI_WAVELENGTH_HETERODYNE);     // GRAY_CODE
    stripe_generator->set_waveLengths(wavelengths);
    stripe_generator->set_phase_shift_number(12);
    stripe_generator->set_A(130);
    stripe_generator->set_B(90);

    // project stripe and capture image
    std::vector<cv::Mat> images_left;
    std::vector<cv::Mat> images_right;

    for (int i = 0; i < stripe_generator->get_pattern_size(); i++)
    {
        std::string filepath_left = "../data/3/L/" + std::to_string(i + 1) + ".bmp";
        cv::Mat image_left = cv::imread(filepath_left, cv::IMREAD_GRAYSCALE);

        std::string filepath_right = "../data/3/R/" + std::to_string(i + 1) + ".bmp";
        cv::Mat image_right = cv::imread(filepath_right, cv::IMREAD_GRAYSCALE);

        if (!image_left.empty() && !image_right.empty())
        {
            images_left.push_back(image_left);
            images_right.push_back(image_right);
        }
    }

    StereoCommon::StereoParameter* stereo_param = new StereoCommon::StereoParameter("../data/3/stereo_params.yaml", images_left[0].size());

    // unwrap phase
    DepthReconstructor* depth_reconstructor = new DepthReconstructor(stripe_generator);
    depth_reconstructor->set_min_B(10);

    cv::Mat phase_image_left;
    depth_reconstructor->phase_reconstruct(images_left, phase_image_left);

    cv::Mat phase_image_left_filtered;
    //noise_filter<double>(phase_image_left, phase_image_left_filtered);
    phase_image_left_filtered = phase_image_left;

    cv::Mat phase_image_right;
    depth_reconstructor->phase_reconstruct(images_right, phase_image_right);

    cv::Mat phase_image_right_filtered;
    //noise_filter<double>(phase_image_right, phase_image_right_filtered);
    phase_image_right_filtered = phase_image_right;

    // depth reconstruct
    cv::Mat disparity;
    depth_reconstructor->set_stereo_param(stereo_param);
    depth_reconstructor->depth_reconstruct(phase_image_left_filtered, phase_image_right_filtered, disparity);

    cv::Mat disparity_filtered;
    //noise_filter<double>(disparity, disparity_filtered, 5);
    disparity_filtered = disparity;

    if (!disparity_filtered.empty())
    {
        // save result
        save_image_to_txt<double>("./disparity.txt", disparity_filtered);
        save_image_to_txt<double>("./left.txt", phase_image_left_filtered);
        save_image_to_txt<double>("./right.txt", phase_image_right_filtered);

        // save point cloud
        std::vector<cv::Vec6f> point_cloud_with_texture;
        convert_disparity_map_to_point_cloud<double>(disparity_filtered, point_cloud_with_texture, stereo_param->Q, cv::Mat(images_left[0].size(), CV_8UC1, 255));

        std::ofstream point_cloud_file("./point_cloud.txt");
        for (auto point : point_cloud_with_texture)
        {
            point_cloud_file << point[0] << " " << point[1] << " " << point[2] << " " << point[3] << " " << point[4] << " " << point[5] << std::endl;
        }
        point_cloud_file.close();

        // show result
        cv::Mat disparity_u8;
        disparity_filtered.convertTo(disparity_u8, CV_8UC1, 0.5);

        cv::imshow("disparity", disparity_u8);
        cv::waitKey(0);
    }
}


int main(int argc, char const *argv[])
{
    //test_multi_wavelength_heterodyne();
    //test_multi_wavelength_heterodyne_from_file();
    //test_stereo_multi_wavelength_heterodyne();
    test_stereo_multi_wavelength_heterodyne_from_file();
    return 0;
}
