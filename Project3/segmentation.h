/*
 * Project 3 Real Time Object 2D Recognition
 *
 * Name:    segmentation.h
 * Author:  Bryan Ang
 * E-mail:  ang.b@northeastern.edu
 * Date:    2023/02/18
 * Purpose: This file contains all methods involving 
			region segmentation and labelling of a
			binary image.
 */

#include <opencv2/core.hpp>

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
	int foregroundValue);