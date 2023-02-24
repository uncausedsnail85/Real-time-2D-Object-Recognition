/*
 * Project 3 Real Time Object 2D Recognition
 *
 * Name:    main.cpp
 * Author:  Bryan Ang
 * E-mail:  ang.b@northeastern.edu
 * Date:    2023/02/16
 * Purpose: This file is the main file of the program. Contains
 *          the logic for the main loop.
 */

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cstring>
#include <cstdio>
#include <string>
#include <cstdlib>

#include "segmentation.h"

// *** Main ***
int main(int argc, char** argv)
{

    // ***** Reading target image and it's feature ***********
    char targetImgPath[256];
    strcpy_s(targetImgPath, argv[1]);
    cv::Mat target = cv::imread(targetImgPath);
    cv::Mat blurTarget;
    cv::GaussianBlur(target, blurTarget, cv::Size(5, 5), 6, 6);
       
    if (target.empty()) // Check for invalid input
    {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    cv::namedWindow("Blurred Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Blurred Image", blurTarget);

    cv::Mat grayscale;
    cv::cvtColor(blurTarget, grayscale, cv::COLOR_BGR2GRAY);
    cv::namedWindow("Grayscale Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Grayscale Image", grayscale);

    //cv::Mat simpleThresholdImg;
    //cv::threshold(grayscale, simpleThresholdImg, 120, 255, cv::THRESH_BINARY);
    //cv::namedWindow("Smple Thresh Image", cv::WINDOW_AUTOSIZE);
    //cv::imshow("Smple Thresh Image", simpleThresholdImg);

    //cv::Mat adaptiveThresholdImg;
    //cv::adaptiveThreshold(grayscale, adaptiveThresholdImg, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 45, 6);
    //cv::namedWindow("Mean Adaptive Thresh Image", cv::WINDOW_AUTOSIZE);
    //cv::imshow("Mean Adaptive Thresh Image", adaptiveThresholdImg);

    cv::Mat adaptiveThresholdGaussianImg;
    cv::adaptiveThreshold(grayscale, adaptiveThresholdGaussianImg, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 45, 6);
    cv::namedWindow("Gauss Adaptive Thresh Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Gauss Adaptive Thresh Image", adaptiveThresholdGaussianImg);

    //cv::Mat otsuThresholdImg;
    //cv::threshold(grayscale, otsuThresholdImg, 0, 255, cv::THRESH_OTSU);
    //cv::namedWindow("Otsu Thresh Image", cv::WINDOW_AUTOSIZE);
    //cv::imshow("Otsu Thresh Image", otsuThresholdImg);

    cv::Mat erodeImg, dilateImg, cleanedImg;
    cv::erode(adaptiveThresholdGaussianImg, erodeImg, cv::Mat(), cv::Point(-1, -1), 1);
    cv::dilate(erodeImg, dilateImg, cv::Mat(), cv::Point(-1, -1), 2);
    cv::erode(dilateImg, erodeImg, cv::Mat(), cv::Point(-1, -1), 1);
    cv::dilate(erodeImg, dilateImg, cv::Mat(), cv::Point(-1, -1), 2);
    cv::erode(dilateImg, cleanedImg, cv::Mat(), cv::Point(-1, -1), 2);
    cv::namedWindow("Cleaned Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Cleaned Image", cleanedImg);

    cv::Mat segmentedImg;
    printf("number of regions: %d", region_growing(cleanedImg, segmentedImg, 255));


    while (true) {
        int userInput = cv::waitKey(0);
        std::cout << userInput;
        if (userInput == 113) { // 113 = q
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}