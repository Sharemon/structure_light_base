/**
 * Copyright @2023 Sharemon. All rights reserved.
 * 
 @author: sharemon
 @date: 2023-10-24
 */

#include <thread>
#include <chrono>

#include "helper.hpp"
#include "../src/DepthReconstrcutor.hpp"
#include "../src/StripeGenerator.hpp"
#include "../src/StereoParameter.hpp"

using namespace StructureLightBase;


void test_multi_wavelength_heterodyne(const Args& args)
{
    // initialize
    Camera *cam = new Camera(0, 1280, 720, -13);
    Projector *projector = new Projector;

    StripeGenerator *stripe_generator = new StripeGenerator(1920, 1080);
    stripe_generator->set_type(args.type);   
    stripe_generator->set_waveLengths(args.wavelengths);
    stripe_generator->set_phase_shift_number(args.phase_shift_number);
    stripe_generator->set_A(args.A);
    stripe_generator->set_B(args.B);

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
    depth_reconstructor->set_min_B(args.Bmin);
    
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


void test_multi_wavelength_heterodyne_from_file(const Args& args)
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
    StripeGenerator* stripe_generator = new StripeGenerator(1920, 1080);
    stripe_generator->set_type(args.type);
    stripe_generator->set_waveLengths(args.wavelengths);
    stripe_generator->set_phase_shift_number(args.phase_shift_number);
    stripe_generator->set_A(args.A);
    stripe_generator->set_B(args.B);

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
    depth_reconstructor->set_min_B(args.Bmin);

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


void test_stereo_multi_wavelength_heterodyne(const Args& args)
{
    // initialize
    Camera *cam = new Camera(0, 2560, 720, -13);
    Projector *projector = new Projector;

    StripeGenerator* stripe_generator = new StripeGenerator(1920, 1080);
    stripe_generator->set_type(args.type);
    stripe_generator->set_waveLengths(args.wavelengths);
    stripe_generator->set_phase_shift_number(args.phase_shift_number);
    stripe_generator->set_A(args.A);
    stripe_generator->set_B(args.B);

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
    depth_reconstructor->set_min_B(args.Bmin);

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


void test_stereo_multi_wavelength_heterodyne_from_file(const Args& args)
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
        std::string filepath_left = "../data/4/L/" + std::to_string(i + 1) + ".bmp";
        cv::Mat image_left = cv::imread(filepath_left, cv::IMREAD_GRAYSCALE);

        std::string filepath_right = "../data/4/R/" + std::to_string(i + 1) + ".bmp";
        cv::Mat image_right = cv::imread(filepath_right, cv::IMREAD_GRAYSCALE);

        if (!image_left.empty() && !image_right.empty())
        {
            image_left.convertTo(image_left, CV_64FC1, 1.0);
            image_right.convertTo(image_right, CV_64FC1, 1.0);

            images_left.push_back(image_left);
            images_right.push_back(image_right);
        }
    }

    StereoCommon::StereoParameter* stereo_param = new StereoCommon::StereoParameter("../data/4/stereo_params.yaml", images_left[0].size());

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


int main(int argc, char *argv[])
{
    Args args;
    parse_args(argc, argv, args);

    //test_multi_wavelength_heterodyne(args);
    //test_multi_wavelength_heterodyne_from_file(args);
    //test_stereo_multi_wavelength_heterodyne(args);
    test_stereo_multi_wavelength_heterodyne_from_file(args);
    return 0;
}
