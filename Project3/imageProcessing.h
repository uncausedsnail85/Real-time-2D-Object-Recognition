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

// Performs the region growing algorithm on a binary image.
// Creates a matrix with region labels on each pixel.
// Params:
//		src: Address of source binary image
//		regionIdImage: Address of region Id Image
//		foregroundValue: value of foreground pixels
//	
// Returns number of regions
int regionGrowing(
	cv::Mat& src,
	cv::Mat& regionIdImage,
	int foregroundValue);

// Finds the largest region from a segmented image.
// Params:
//	src: Address of source binary image
//  regionIdImage: Address of region Id Image
//  filteredImage: The address to return the filtered image,
//				   where only the largest region remains.
//  numOfRegions: number of regions to search through
// 
// Returns the ID of the region.
int filterOnlylargestRegion(
	cv::Mat& src,
	cv::Mat& regionIdImage,
	cv::Mat& filteredImage,
	int numOfRegions
);

// Finds the axis of least central moment and
// returns the moment around that axis.
// angle is the output for the angle of the axis
double momentAroundCentralAxis(cv::Mat& src, int foregroundValue, double& angle);

// draw axis lines and bounding box on an image
// Params:
//	src:	Address of source binary image. Image should
//			be cleaned up and have only one region
//  output: Output matrix
// 
// Returns status of run.
int drawAxisLinesAndBoundingBox(const cv::Mat& src, cv::Mat& output);



