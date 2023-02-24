/*
 * Project 3 Real Time Object 2D Recognition
 *
 * Name:    BinaryProcessing.cpp
 * Author:  Bryan Ang
 * E-mail:  ang.b@northeastern.edu
 * Date:    2023/02/16
 * Purpose: This file contains all methods involving the
			processing/cleanup of a binary image,
			which are any erosion and growing methods.
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

int growBinaryImage(cv::Mat& src, cv::Mat& dst, int iteraions, bool background, bool connectedness) {

	dst = cv::Mat::zeros(src.size(), CV_8UC3);

	for (int i = 0; i < src.rows; i++) {
		cv::Vec3b* srcRptrm1 = i == 0 ? src.ptr<cv::Vec3b>(i) : src.ptr<cv::Vec3b>(i - 1);
		cv::Vec3b* srcRptr = src.ptr<cv::Vec3b>(i);
		cv::Vec3b* dstRptr = dst.ptr<cv::Vec3b>(i);
		for (int j = 0; j < src.cols; j++) {

			//4-con
			schar neighbors[2];



			
		}
	}
}