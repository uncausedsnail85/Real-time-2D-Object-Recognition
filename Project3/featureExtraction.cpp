/*
 * Project 3 Real Time Object 2D Recognition
 *
 * Name:    imageProcessing.h
 * Author:  Bryan Ang
 * E-mail:  ang.b@northeastern.edu
 * Date:    2023/02/24
 * Purpose: This file contains all methods involving
			features of an image. Includes feature
			calculations, saving and read of the DB.
			(Task 5 - 9)
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
#include <map>
#include <set>

#include "featureExtraction.h"
#include "imageProcessing.h"

// Function to read db file and return a vector of all feature vectors + label vector
int readDBFile(std::string filename,
	std::vector<std::vector<double>>& allFeatures,
	std::vector<std::string>& labels);

// Generates a vector of features
// {percetangeFilled, h/w ratio, moment around axis of least central moment}
// Input is a binary image with only 1 region
int getFeatures(const cv::Mat& src, std::vector<double>& featureVector) {

	// reset vector
	featureVector.clear();

	// Moment around central axis 
	cv::Moments moments = cv::moments(src, true);
	double hu[7];
	cv::HuMoments(moments, hu);

	// find rotated bounding box points	
	std::vector<cv::Point> regionPoints;
	cv::findNonZero(src, regionPoints);
	cv::RotatedRect rotatedBB = cv::minAreaRect(regionPoints);

	// calculate the area of the rotated bounding box
	double width = rotatedBB.size.width;
	double height = rotatedBB.size.height;
	double theta = rotatedBB.angle * CV_PI / 180.0; // convert angle from degrees to radians
	if (theta < -CV_PI / 2) { // adjust angle to be in range (-pi/2, pi/2)
		theta += CV_PI;
	}
	else if (theta > CV_PI / 2) {
		theta -= CV_PI;
	}
	double rotatedWidth = fabs(width * cos(theta)) + fabs(height * sin(theta));
	double rotatedHeight = fabs(width * sin(theta)) + fabs(height * cos(theta));
	double rotatedArea = rotatedWidth * rotatedHeight;

	// h/w ratio
	double hwRatio = std::max(rotatedHeight, rotatedWidth) / std::min(rotatedHeight, rotatedWidth);
	// percentage filled
	int regionArea = cv::countNonZero(src);
	double percentFilled = static_cast<double>(regionArea) / rotatedArea * 100.0;

	// add to vector
	featureVector.push_back(percentFilled);
	featureVector.push_back(hwRatio);
	for (int i = 0; i < 7; i++) {
		featureVector.push_back(hu[i]);
	}
	return 1;
}

// writes a label and its features to a file
int writeFeaturesToFile(std::vector<double> features, std::string label, std::string filename)
{
	std::ofstream outfile(filename, std::ios::app); // open file for writing, appending to existing data if any
	if (!outfile.is_open()) {
		std::cerr << "unable to open db file (write)." << std::endl;
		return 0;
	}
	// iterate over feature vector and write each double to file, separated by a space
	for (const double& feature : features) {
		outfile << feature << " ";
	}
	// write label to file
	outfile << label << " " << std::endl;
	outfile.close(); // close the file
	printf("Saved %s to file.\n", label.c_str());

	return 1;
}	

// Function to read db file and return a vector of all feature vectors + label vector
int readDBFile(std::string filename,
	std::vector<std::vector<double>>& allFeatures,
	std::vector<std::string>& labels)
{
	std::ifstream infile(filename);
	if (!infile.is_open()) {
		std::cerr << "unable to open db file (read)." << std::endl;
		return 0;
	}

	// read file in a vector of feature vectors
	std::string line;
	allFeatures.clear();
	labels.clear();
	while (std::getline(infile, line)) {
		// read line
		std::istringstream iss(line);
		// create feature vector
		std::vector<double> features;
		// var to hold a feature
		double feature;
		for (int i = 0; i < 9; i++) {
			iss >> feature;
			features.push_back(feature);
		}
		allFeatures.push_back(features);

		// read label at end of line
		std::string label;
		iss >> label;
		labels.push_back(label);
	}
	infile.close();
	return 1;
}

// calculates standard deviation of each feature in a file
int getStandardDeviation(std::vector<std::vector<double>> allFeatures, std::vector<double> &stdDeviations) {

	// reset stdDeviations vector
	stdDeviations.clear();

	// find mean
	std::vector<double> means;
	// for each feature in a feature vector
	for (int featureIdx = 0; featureIdx < allFeatures[0].size(); featureIdx++) {
		double sum = 0, mean = 0;
		// for each feature vector in all features
		for (int allFeaturesIdx = 0; allFeaturesIdx < allFeatures.size(); allFeaturesIdx++) {
			sum = sum + allFeatures[allFeaturesIdx][featureIdx];
		}
		mean = sum / allFeatures.size();
		means.push_back(mean);
	}

	// find std dev
	// for each feature in a feature vector
	for (int featureIdx = 0; featureIdx < allFeatures[0].size(); featureIdx++) {
		double sum = 0;
		// for each feature vector in all features
		for (int allFeaturesIdx = 0; allFeaturesIdx < allFeatures.size(); allFeaturesIdx++) {
			double diff = allFeatures[allFeaturesIdx][featureIdx] - means[featureIdx];
			sum = sum + diff*diff;
		}
		double stdDev = sqrt(sum / allFeatures.size());
		stdDeviations.push_back(stdDev);
	}
	//for (int i = 0; i < stdDeviations.size(); i++) {
	//	printf("mean[%d]: %f; ", i, means[i]);
	//}
	//printf("\n");
	//for (int i = 0; i < stdDeviations.size(); i++) {
	//	printf("Stddev[%d]: %f; ", i, stdDeviations[i]);
	//}
	//printf("\n");
	//for (int i = 0; i < labels.size(); i++) {
	//	printf("labels[%d]: %s; ", i, labels[i].c_str());
	//}
	return 1;	
}

// Closet neighbor classifer
// Given a feature vector, find the label with the lowest
// cumalative distance between features (scaled euclidean).
// Output is a string
// Returns the row index of the nearest neighbor in db
int nearestNeigborDistance(std::vector<double> targetFeatures,
	std::string filename,
	std::string& outputLabel)
{

	std::vector<std::vector<double>> allFeatures;
	std::vector<std::string> labels;
	bool readStatus = 0;
	readStatus = readDBFile(filename, allFeatures, labels);
	if (!readStatus) {
		outputLabel = "no db file";
		return -1;
	}

	// edge cases
	if (labels.size() == 1) {
		outputLabel = labels[0];
		return 1;
	}
	else if (labels.size() == 0) {
		outputLabel = "No data in db file";
		return 0;
	}

	// get std dev
	std::vector<double> stdDeviations;
	getStandardDeviation(allFeatures, stdDeviations);

	double minDistance = DBL_MAX;
	int nearestNeigborIndex = -1;
	// for all data points in db
	for (int i = 0; i < allFeatures.size(); i++) {

		// calculate distance to data point
		double distanceSum = 0;
		// for each feature
		for (int j = 0; j < allFeatures[i].size(); j++) {
			double distanceScaled = (targetFeatures[j] - allFeatures[i][j])/ stdDeviations[j];
			double distanceSquared = distanceScaled * distanceScaled;
			distanceSum = distanceSum + distanceSquared;
		} // for features	

		// if distance to data point < minDistance
		if (distanceSum < minDistance) {
			nearestNeigborIndex = i;
			minDistance = distanceSum;
		} 	
	} // for db loop

	outputLabel = labels[nearestNeigborIndex];
	return nearestNeigborIndex;
}

// k-nearest neighbor classifer
// Given a feature vector, find the label with the lowest mean
// distance between features (scaled euclidean).
// 
// Calculates distance to every other label, and finds lowest
// k sum.
// 
// Output is a string
// Returns the row index of the nearest neighbor in db
int kNearestNeigborDistance(std::vector<double> targetFeatures,
	std::string filename,
	int k,
	float std_multiplier,
	std::string& outputLabel)
{	
	// get all features and labels
	std::vector<std::vector<double>> allFeatures;
	std::vector<std::string> labels;
	bool readStatus = 0;
	readStatus = readDBFile(filename, allFeatures, labels);
	if (!readStatus) {
		outputLabel = "no db file";
		return -1;
	}

	// edge cases
	if (labels.size() == 1) {
		outputLabel = labels[0];
		return 1;
	}
	else if (labels.size() == 0) {
		outputLabel = "No data in db file";
		return 0;
	}

	// get std dev
	std::vector<double> stdDeviations;
	getStandardDeviation(allFeatures, stdDeviations);

	// turn labels in a set
	std::set<std::string> setOfLabels;
	for (int i = 0; i < labels.size(); i++) {
		setOfLabels.insert(labels[i]);
	}

	double minDistance = DBL_MAX;
	// for every label
	for (std::string label : setOfLabels) {
		// create vector of distances to all labels
		std::vector<double> distances;

		// for every feature vector in db
		for (int i = 0; i < allFeatures.size(); i++) {

			// if label is label we want
			if (labels[i] == label) {
				// calculate the distance to that data point 
				// (which is the same label as we want)
				double distanceSum = 0;
				// for each feature
				for (int j = 0; j < allFeatures[i].size(); j++) {
					double distanceScaled = (targetFeatures[j] - allFeatures[i][j]) / stdDeviations[j];
					double distanceSquared = distanceScaled * distanceScaled;
					distanceSum = distanceSum + distanceSquared;
				} // for each feature

				// add distance to that data point to vector of distances
				distances.push_back(distanceSum);
			}
		} // for db loop

		// once we have all distances to this label:
		// sort vector and take top k
		std::sort(distances.begin(), distances.end());
		// sum top k
		double finalDistance = 0;
		for (int i = 0; i < k; i++) {
			finalDistance = finalDistance + distances[i];
		}

		// take min final distances amoung all labels
		if (finalDistance < minDistance) {
			minDistance = finalDistance;
			outputLabel = label;
		} // if
	} // for each label


	/* 
	* THRESHOLDING
	* Using the min distance / sum of stdDev
	* Because min distance is a sum of feature distances
	*/
	// Check if distance to nearest neighbor is within std_multiplier standard deviations sum
	double sumOfStdDev =  0;
	for (int i = 0; i < stdDeviations.size(); i++) {
		sumOfStdDev = sumOfStdDev + stdDeviations[i];
	}
	float dist_stddev = minDistance / sumOfStdDev;
	if (dist_stddev > std_multiplier) {
		// distance to nearest neighbor is above std_multiplier standard deviations, so prediction is unreliable
		outputLabel = "Unkown"; 
		return -1;
	}

} // func kNearestNeigborDistance()