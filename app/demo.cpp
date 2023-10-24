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
        cv::moveWindow("projector", 0, 0);
        cv::setWindowProperty("projector", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        cv::waitKey(1);
    }
};

using namespace StructureLightBase;

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
    stripe_generator->set_phase_shift_number(4);
    stripe_generator->set_A(128);
    stripe_generator->set_B(90);

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
        cv::imwrite("./" + std::to_string(i) + ".png", pattern);

        //projector->project(pattern);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        if (cam->capture(image))
        {
            images.push_back(pattern);
        }
        else
        {
            std::cout << "cannot read " << i << "th image." << std::endl;
        }
    }


    // unwrap phase and reconstruct depth
    DepthReconstructor *depth_reconstructor = new DepthReconstructor(stripe_generator);
    depth_reconstructor->set_min_B(50);
    
    cv::Mat phase_image;
    depth_reconstructor->reconstruct(images, phase_image);

    if (!phase_image.empty())
    {
        // save result
        save_image_to_txt<double>("./result.txt", phase_image);

        // show result
        cv::Mat phase_image_u8;
        phase_image.convertTo(phase_image_u8, CV_8UC1, 255);

        cv::imshow("phase unwrap", phase_image_u8);
        cv::waitKey(0);
    }
}




int main(int argc, char const *argv[])
{
    test_multi_wavelength_heterodyne();

    return 0;
}
