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

#include "imageProcessing.h"
#include "driverFunctions.h"
#include "featureExtraction.h"

// *** Main ***
int main(int argc, char** argv)
{
    executeVideoFeed();

    return 0;
}
