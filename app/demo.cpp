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
void noise_filter(const cv::Mat&src, cv::Mat& dst)
{
    int w = src.cols;
    int h = src.rows;

    const int ksize = 5;
    const T value_threshold = 0.05;
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

void test_multi_wavelength_heterodyne_from_file()
{
    // initialize
#if 0
    std::vector<int> wavelengths = { 28, 26, 24 };
    StripeGenerator* stripe_generator = new StripeGenerator(2184, 1536);
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
    depth_reconstructor->reconstruct(images, phase_image);

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
    depth_reconstructor->reconstruct(images, phase_image);

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




int main(int argc, char const *argv[])
{
    //test_multi_wavelength_heterodyne();
    test_multi_wavelength_heterodyne_from_file();
    return 0;
}
