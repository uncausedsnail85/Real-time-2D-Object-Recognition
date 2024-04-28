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

 // Generates a vector of features
 // {percetangeFilled, h/w ratio, moment around axis of least central moment}
 // Input is a binary image with only 1 region
int getFeatures(const cv::Mat& src, std::vector<double>& featureVector);

// writes a label and its features to a file
int writeFeaturesToFile(std::vector<double> features,
	std::string label,
	std::string filename);

// Function to read db file and return a vector of all feature vectors + label vector
int readDBFile(std::string filename, 
	std::vector<std::vector<double>>& allFeatures,
	std::vector<std::string>& labels);

// calculates standard deviation of each feature from a vector of feature vectors
int getStandardDeviation(std::vector<std::vector<double>> allFeatures,
	std::vector<double>& stdDeviations);

// Closet neighbor classifer
// Given a feature vector, find the label with the lowest
// cumalative distance between features (scaled euclidean).
// Output is a string
// Returns the row index of the nearest neighbor in db
int nearestNeigborDistance(std::vector<double> targetFeatures,
	std::string filename,
	std::string& outputLabel);

// k-nearest neighbor classifer
// Given a feature vector, find the label with the lowest mean
// distance between features (scaled euclidean).
// Output is a string
// Returns the row index of the nearest neighbor in db
int kNearestNeigborDistance(std::vector<double> targetFeatures,
	std::string filename,
	int k,
	float std_multiplier,
	std::string& outputLabel);