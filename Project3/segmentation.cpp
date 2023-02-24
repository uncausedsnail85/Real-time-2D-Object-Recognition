/*
 * Project 3 Real Time Object 2D Recognition
 *
 * Name:    segmentation.cpp
 * Author:  Bryan Ang
 * E-mail:  ang.b@northeastern.edu
 * Date:    2023/02/18
 * Purpose: This file contains all methods involving
			region segmentation and labelling of a
			binary image.
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

#include "segmentation.h"

// Performs the region growing algorithm on a binary image.
// Creates a matrix with region labels on each pixel.
// Params:
//		src: Address of source binary image
//		regionIdImage: Address of region Id Image
//		foregroundValue: value of foreground pixels
//		backgroundValue: value of background pixels
//		connectedness: whether 4 or 8 connected. 0 or 1
//	
// Returns number of regions
int region_growing(
	cv::Mat& src,
	cv::Mat& regionIdImage,
	int foregroundValue)
{
	// create stack. each element in stack is
	// address of a pixel
	// std::stack<uchar*> stack;
	std::stack<std::pair<int, int>> stack;
	int currId = 1;
	int numOfRegions = 0;
	regionIdImage = cv::Mat::zeros(src.size(), CV_8UC3);

	// for each pixel
	for (int i = 0; i < src.rows; i++) {
		cv::Vec3b* srcRptr = src.ptr<cv::Vec3b>(i);
		cv::Vec3b* regionRptr = regionIdImage.ptr<cv::Vec3b>(i);
		for (int j = 0; j < src.cols; j++) {

			// if pixel is foreground and unlabelled
			if (srcRptr[j][0] == foregroundValue && regionRptr[j][0] == 0) {
				regionRptr[j][0] == currId;
				stack.push(std::make_pair(i, j));
				while (!stack.empty()) {
					int i = stack.top().first;
					int j = stack.top().second;
					stack.pop();
					// FOR EACH neighbor of pixel 4-con
					if (src.at<uchar>(i - 1, j) == foregroundValue
						&& regionIdImage.at<uchar>(i - 1, j) == 0) {
						src.at<uchar>(i - 1, j) = currId;
						stack.push(std::make_pair(i - 1, j));
					} // if upper neighbor
					if (src.at<uchar>(i, j - 1) == foregroundValue
						&& regionIdImage.at<uchar>(i, j - 1) == 0) {
						src.at<uchar>(i, j - 1) = currId;
						stack.push(std::make_pair(i, j - 1));
					} // if left neighbor
				} // while stack not empty
				currId++;
			} // if pixel in fg and unlabelled
		} // for cols
	}// for rows
}