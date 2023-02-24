/*
 * Project 3 Real Time Object 2D Recognition
 *
 * Name:    BinaryProcessing.h
 * Author:  Bryan Ang
 * E-mail:  ang.b@northeastern.edu
 * Date:    2023/02/16
 * Purpose: This file contains all methods involving the
			processing/cleanup of a binary image,
			which are any erosion and growing methods.
 */

#include <opencv2/core.hpp>

// Grows a given binary image.
// Params:
//		src: Address of source image
//		dst: Address of processed image
//		iteraions: Number of iteraions to perform.
//		background: whether 0 or 255 is the background
//					value is either 0 or 1.
//		connectedness: whether 4 or 8 connected. 0 or 1
//					   respectively
int growBinaryImage(cv::Mat& src, cv::Mat& dst, int iteraions, bool background, bool connectedness);

// Erodes a given binary image.
// Params:
//		src: Address of source image
//		dst: Address of processed image
//		iteraions: Number of iteraions to perform.
//		background: whether 0 or 255 is the background
//					value is either 0 or 1.
int erodeBinaryImage(cv::Mat& src, cv::Mat& dst, int iteraions, bool background, bool connectedness);

// Performs distance transformation on an image using
// the Grassfire Transform.
// Helper function.
// Params:
//		src: Address of source image
//		dst: Address of processed image
//		background: whether 0 or 255 is the background
//					value is either 0 or 1.
int grassfireTransform(cv::Mat& src, cv::Mat& dst, bool background, bool connectedness);

