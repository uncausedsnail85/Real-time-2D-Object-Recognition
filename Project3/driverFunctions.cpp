/*
 * Project 3 Real Time Object 2D Recognition
 *
 * Name:    driverFunctions.cpp
 * Author:  Bryan Ang
 * E-mail:  ang.b@northeastern.edu
 * Date:    2023/02/24
 * Purpose: This file contains all methods involving
			running the main logic for pipelines.
			I.e. running the video display loop
			or single image runs.
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
#include <stack>
#include <queue>
#include <math.h>

#include "imageProcessing.h"
#include "featureExtraction.h"

 // executes the pipeline for live video feed
int executeVideoFeed() {
    cv::VideoCapture* capdev;
    // open the video device
    capdev = new cv::VideoCapture(0);
    if (!capdev->isOpened()) {
        printf("Unable to open video device\n");
        return(-1);
    }
    // get some properties of the image
    cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::namedWindow("Video", 1); // create a window

    // init variables for loop
    cv::Mat frame;
    cv::Mat displayFrame;
    int modifierFlag = 0;
    std::string label;

    // video loop
    for (;;) {
        *capdev >> frame; // get a new frame from the camera, treat as a stream

        // Process Image:
        // blur
        cv::Mat blurTarget;
        cv::GaussianBlur(frame, blurTarget, cv::Size(5, 5), 6, 6);

        // grey
        cv::Mat grayscale;
        cv::cvtColor(blurTarget, grayscale, cv::COLOR_BGR2GRAY);
       
        // threshold
        cv::Mat otsuThresholdImg, finalThresholdImg;
        cv::threshold(grayscale, otsuThresholdImg, 0, 255, cv::THRESH_OTSU);
        cv::bitwise_not(otsuThresholdImg, finalThresholdImg);
        //cv::adaptiveThreshold(grayscale, finalThresholdImg, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 40, 6);
        
        // clean up
        cv::Mat erodeImg, dilateImg, cleanedImg;
        //cv::erode(finalThresholdImg, erodeImg, cv::Mat(), cv::Point(-1, -1), 1);
        cv::dilate(finalThresholdImg, dilateImg, cv::Mat(), cv::Point(-1, -1), 4);
        cv::erode(dilateImg, erodeImg, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(erodeImg, dilateImg, cv::Mat(), cv::Point(-1, -1), 4);
        cv::erode(dilateImg, cleanedImg, cv::Mat(), cv::Point(-1, -1), 6);
        
        // create labelMap
        cv::Mat labelMap;
        int numOfRegions = regionGrowing(cleanedImg, labelMap, 255);
       
        // retain only largest region in image
        cv::Mat largestRegionImage;
        int largestRegionId = filterOnlylargestRegion(cleanedImg, labelMap, largestRegionImage, numOfRegions);
        
        // compute features
        std::vector<double> featureVector;
        getFeatures(largestRegionImage, featureVector);

        // find nearest neighbor label
        std::string nnLabel;
        //int nnIndex = nearestNeigborDistance(featureVector, "db.txt", nnLabel);
        // using k-nearest neighbor, k =2
        int nnIndex = kNearestNeigborDistance(featureVector, "db.txt", 2, 1,  nnLabel);
 
        // whether to display raw image or processed image
        switch (modifierFlag) {
            case 0: // raw
                frame.copyTo(displayFrame);
                break;
            case 1: // final
                largestRegionImage.copyTo(displayFrame);
                drawAxisLinesAndBoundingBox(largestRegionImage, displayFrame);
                break;
            case 2: // threshold
                finalThresholdImg.copyTo(displayFrame);
                break;
            case 3: // cleaned threshold
                cleanedImg.copyTo(displayFrame);
                break;
            case 4: // region map
                // Create color image for visualization
                cv::Mat regionImage;
                cv::cvtColor(cleanedImg, regionImage, cv::COLOR_GRAY2BGR);  
                // Loop over regions and color each one differently
                for (int i = 1; i < numOfRegions; i++)
                {
                    // Create a mask for this region
                    // a mask is an image where it is 0 when false, 255 when true
                    // in this case true when at i (label id) for labelImg
                    cv::Mat mask = labelMap == i;

                    // Generate a random color for this region
                    cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);

                    // Apply the color to the region in the color image
                    regionImage.setTo(color, mask);
                }
                regionImage.copyTo(displayFrame);
                break;
        }
        // draw label    
        cv::putText(displayFrame, nnLabel, cv::Point(30, 30),
            cv::FONT_HERSHEY_DUPLEX, 1.1,
            cv::Scalar(255, 255, 255));
                
        // show display frame
        cv::imshow("Video", displayFrame);
        
        // show display frame
        cv::imshow("Video", displayFrame);
        
        char key = cv::waitKey(10);
        if (key == 'q') {
            break;
        }
        else if (key == 'd') {
            modifierFlag = modifierFlag == 1 ? 0 : 1;
        }
        else if (key == ' ') {
            std::cout << "Enter label: ";
            std::getline(std::cin, label);
            std::cout << "Label: " << label << std::endl;
            writeFeaturesToFile(featureVector, label, "db.txt");            
        }
        else if (key == 't') {
            modifierFlag = modifierFlag == 2 ? 0 : 2;
        }
        else if (key == 'c') {
            modifierFlag = modifierFlag == 3 ? 0 : 3;
        }
        else if (key == 'r') {
            modifierFlag = modifierFlag == 4 ? 0 : 4;
        }
    }
    delete capdev;
    return(0);
}

// Executes the pipeline for a single specified image
// For development purposes
int executeSingleImage(cv::Mat& src) {

    cv::Mat blurTarget;
    cv::GaussianBlur(src, blurTarget, cv::Size(5, 5), 6, 6);

    cv::namedWindow("Blurred Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Blurred Image", blurTarget);

    cv::Mat grayscale;
    cv::cvtColor(blurTarget, grayscale, cv::COLOR_BGR2GRAY);
    cv::namedWindow("Grayscale Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Grayscale Image", grayscale);

    //cv::Mat simpleThresholdImg;
    //cv::threshold(grayscale, simpleThresholdImg, 90, 255, cv::THRESH_BINARY_INV);
    //cv::namedWindow("Smple Thresh Image", cv::WINDOW_AUTOSIZE);
    //cv::imshow("Smple Thresh Image", simpleThresholdImg);

    //cv::Mat adaptiveThresholdImg;
    //cv::adaptiveThreshold(grayscale, adaptiveThresholdImg, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 45, 6);
    //cv::namedWindow("Mean Adaptive Thresh Image", cv::WINDOW_AUTOSIZE);
    //cv::imshow("Mean Adaptive Thresh Image", adaptiveThresholdImg);

    //cv::Mat adaptiveThresholdGaussianImg;
    //cv::adaptiveThreshold(grayscale, adaptiveThresholdGaussianImg, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 45, 6);
    //cv::namedWindow("Gauss Adaptive Thresh Image", cv::WINDOW_AUTOSIZE);
    //cv::imshow("Gauss Adaptive Thresh Image", adaptiveThresholdGaussianImg);

    cv::Mat otsuThresholdImg, otsuThresholdImgInv;
    cv::threshold(grayscale, otsuThresholdImg, 0, 255, cv::THRESH_OTSU);
    cv::bitwise_not(otsuThresholdImg, otsuThresholdImgInv);
    cv::namedWindow("Otsu Thresh Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Otsu Thresh Image", otsuThresholdImgInv);

    cv::Mat erodeImg, dilateImg, cleanedImg;
    cv::erode(otsuThresholdImgInv, erodeImg, cv::Mat(), cv::Point(-1, -1), 1);
    cv::dilate(erodeImg, dilateImg, cv::Mat(), cv::Point(-1, -1), 2);
    cv::erode(dilateImg, erodeImg, cv::Mat(), cv::Point(-1, -1), 1);
    cv::dilate(erodeImg, dilateImg, cv::Mat(), cv::Point(-1, -1), 2);
    cv::erode(dilateImg, cleanedImg, cv::Mat(), cv::Point(-1, -1), 2);
    cv::namedWindow("Cleaned Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Cleaned Image", cleanedImg);

    cv::Mat labelMap;
    int numOfRegions = regionGrowing(cleanedImg, labelMap, 255);
    //int numOfRegions = cv::connectedComponents(cleanedImg, labelImg);

    printf("number of regions: %d\n", numOfRegions);

    // Create color image for visualization
    cv::Mat regionImage;
    cv::cvtColor(cleanedImg, regionImage, cv::COLOR_GRAY2BGR);

    // Loop over regions and color each one differently
    for (int i = 1; i < numOfRegions; i++)
    {
        // Create a mask for this region
        // a mask is an image where it is 0 when false, 255 when true
        // in this case true when at i (label id) for labelImg
        cv::Mat mask = labelMap == i;

        // Generate a random color for this region
        cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);

        // Apply the color to the region in the color image
        regionImage.setTo(color, mask);
    }
    cv::namedWindow("Region Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Region Image", regionImage);

    cv::Mat largestRegionImage;
    int areaOfLargestRegion;
    int largestRegionId = filterOnlylargestRegion(cleanedImg, labelMap, largestRegionImage, numOfRegions);
    cv::namedWindow("largest region", cv::WINDOW_AUTOSIZE);
    cv::imshow("largest region", largestRegionImage);

    //momentAroundCentralAxis(largestRegionImage, 255);

    while (true) {
        int userInput = cv::waitKey(0);
        std::cout << userInput;
        if (userInput == 113) { // 113 = q
            break;
        }
    }
    cv::destroyAllWindows();
    return 1;
}
