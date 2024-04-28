/*
 * Project 3 Real Time Object 2D Recognition
 *
 * Name:    imageProcessing.h
 * Author:  Bryan Ang
 * E-mail:  ang.b@northeastern.edu
 * Date:    2023/02/18
 * Purpose: This file contains all methods involving
			any custom functions that process the image
			before feature extraction.
			Such as segmentation, filtering regions,
			and the drawing axis and boundary box.
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
#include <fstream>

#include "imageProcessing.h"

 // Performs the region growing algorithm on a binary image.
 // Creates a matrix with region labels on each pixel.
 // Params:
 //		src: Address of source binary image
 //		regionIdImage: Address of region Id Image
 //		foregroundValue: value of foreground pixels
 //	
 // Returns number of regions
int regionGrowing(cv::Mat& src, cv::Mat& regionIdImage, int foregroundValue)
{
	// init queue, label, regionIdImage
	std::queue<cv::Point> q;
	int currId = 1;
    regionIdImage = cv::Mat::zeros(src.size(), CV_32SC1);

    // init offset to get neighbors
	// TODO: Parameterize connectedness
	// 4-con
    //int dx[4] = {0, -1, 1, 0};
    //int dy[4] = {-1, 0, 0, 1};
    //int numNeighbors = 4;
	// 8-con
	int dx[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
	int dy[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
	int numNeighbors = 8;

    // for each pixel
    for (int i = 0; i < src.rows; i++) {
		//cv::Vec3b* srcRptr = src.ptr<cv::Vec3b>(i);
		//cv::Vec3b* regionRptr = regionIdImage.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            // if pixel is foreground and unlablled
            if (src.at<uchar>(i, j) == foregroundValue && regionIdImage.at<int>(i, j) == 0) {
                q.push(cv::Point(j, i));
				// label
                regionIdImage.at<int>(i, j) = currId;

                // while queue not empty (go over connected neigbors)
                while (!q.empty()) {
                    cv::Point point = q.front();
                    q.pop();

                    // for each neighbors of this pixel
                    for (int k = 0; k < numNeighbors; k++) {
                        // Compute coordinates of neighbor pixel
                        int neighborX = point.x + dx[k];
                        int neighborY = point.y + dy[k];

                        // bounds check
                        if (neighborX >= 0 
							&& neighborY >= 0 
							&& neighborX < src.cols 
							&& neighborY < src.rows)
                        {
                            // If neighbor is foreground and unmarked, add to queue and mark
                            if (src.at<uchar>(neighborY, neighborX) == foregroundValue 
								&& regionIdImage.at<int>(neighborY, neighborX) == 0)
                            {
                                q.push(cv::Point(neighborX, neighborY));
                                regionIdImage.at<int>(neighborY, neighborX) = currId;
                            } // if neighbor is fg and unlabelled
                        } // if bounds check
                    } // for neighbors
                } // while queue
                currId++;
            } // if pixel is fg and unlabelled
        } // for cols
    } // for rows
    return currId - 1;
}

// Finds the largest region from a segmented image.
// Params:
//	src: Address of source binary image
//  regionIdImage: Address of region Id Image
//  largestRegionMask: Address of a matrix containing the largest region (Output)
//  numOfRegions: number of regions to search through
// 
// Returns the ID of the region.
int filterOnlylargestRegion(cv::Mat& src, cv::Mat& regionIdImage, cv::Mat& largestRegionMask, int numOfRegions)
{
	int largestLabel = 0;
	int largestArea = 0;
	for (int i = 1; i <= numOfRegions; i++)
	{
		// create binary mask for current region
		cv::Mat mask = regionIdImage == i;

		// calculate area of region
		int area = cv::countNonZero(mask);

		// check if region is larger than previous largest region
		if (area > largestArea) {
			largestLabel = i;
			largestArea = area;
		}
	}
	// create binary mask for largest component
	largestRegionMask = regionIdImage == largestLabel;
	return largestLabel;
}

// Finds the axis of least central moment and
// returns the moment around that axis.
// angle is the output for the angle of the axis
double momentAroundCentralAxis(cv::Mat& src, int foregroundValue, double& angle) {
	cv::Moments moments = cv::moments(src, true);

	double cx = moments.m10 / moments.m00; // origin
	double cy = moments.m01 / moments.m00; // origin

	// hu moments?

	double mu02 = 0, mu20 = 0, mu11 = 0;
	// for each pixel
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			// if part of region, f(x,y)
			if (src.at<uchar>(i, j) == foregroundValue) {
				mu02 = mu02 + (i - cy) * (i - cy); // row, y
				mu20 = mu20 + (j - cx) * (j - cx); // col, x
				mu11 = mu11 + (i - cy) * (j - cx);
			}
		} // for col
	} // for row
	mu02 = mu02 / moments.m00;
	mu20 = mu20 / moments.m00;
	mu11 = mu11 / moments.m00;

	// angle of axis of least moment
	double alpha = 0.5 * atan(2 * mu11 / (mu20 - mu02));

	// append central moments into feature vector
	
	double beta = alpha + CV_PI / 2; // alpha + pi / 2
	double mu22alpha = 0;
	// for each pixel
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			// if part of region, f(x,y)
			if (src.at<uchar>(i, j) == foregroundValue) {
				double additional = (i - cy) * cos(beta) + (j + cx) * sin(beta);
				mu22alpha = mu22alpha + additional*additional;
			}
		} // for col
	} // for row
	mu22alpha = mu22alpha / moments.m00;
	angle = alpha;
	return mu22alpha;
}

// draw axis lines and bounding box on an image
int drawAxisLinesAndBoundingBox(const cv::Mat& src, cv::Mat& output) {
	cv::Moments moments = cv::moments(src, true);

	// centers
	double cx = moments.m10 / moments.m00; // origin
	double cy = moments.m01 / moments.m00; // origin

	// hu moments?

	double mu02 = 0, mu20 = 0, mu11 = 0;
	// for each pixel
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			// if part of region, f(x,y)
			if (src.at<uchar>(i, j) == 255) {
				mu02 = mu02 + (i - cy) * (i - cy); // row, y
				mu20 = mu20 + (j - cx) * (j - cx); // col, x
				mu11 = mu11 + (i - cy) * (j - cx);
			}
		} // for col
	} // for row
	mu02 = mu02 / moments.m00;
	mu20 = mu20 / moments.m00;
	mu11 = mu11 / moments.m00;

	// angle of axis of least moment
	double tempD = mu20 < mu02 ? CV_PI / 2 : 0;
	double alpha = 0.5 * atan(2 * mu11 / (mu20 - mu02)) + tempD;
	double beta = alpha + CV_PI / 2; // alpha + pi / 2
	// length of line to draw
	double l = 300;

	// find points on line
	cv::Point axisOrigin = cv::Point(cx, cy);
	cv::Point xAxisEnd = cv::Point(cx + l * cos(alpha), cy + l * sin(alpha));
	//double yAxisAngle = (CV_PI / 2) - alpha;
	//cv::Point yAxisEnd = cv::Point(cx + l * cos(yAxisAngle), cy + l * sin(yAxisAngle));

	// draw line
	cv::Mat temp = cv::Mat::zeros(src.size(), CV_8UC3);
	cv::cvtColor(src, temp, cv::COLOR_GRAY2BGR);
	cv::line(temp, axisOrigin, xAxisEnd, cv::Scalar(255, 0, 255), 2);
	//cv::line(temp, axisOrigin, yAxisEnd, cv::Scalar(255, 255, 0), 2);

	/* 
	// find bounding box
	cv::Mat tempLabels, stats, centroids;
	cv::connectedComponentsWithStats(src, tempLabels, stats, centroids); // input image should be 1
	int leftmostX = stats.at<int>(cv::Point(cv::CC_STAT_LEFT, 1));
	int topmostY = stats.at<int>(cv::Point(cv::CC_STAT_TOP, 1));
	int width = stats.at<int>(cv::Point(cv::CC_STAT_WIDTH, 1));
	int height = stats.at<int>(cv::Point(cv::CC_STAT_HEIGHT, 1));
	*/

	// find rotated bounding box points	
	std::vector<cv::Point> regionPoints;
	cv::findNonZero(src, regionPoints);
	cv::RotatedRect rotatedBB = cv::minAreaRect(regionPoints);
	cv::Point2f rect_points[4];
	rotatedBB.points(rect_points);
	// draw box with lines
	for (int j = 0; j < 4; j++) {
		line(temp, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(255, 0, 255));
	}

	temp.copyTo(output);
	return 1;
}