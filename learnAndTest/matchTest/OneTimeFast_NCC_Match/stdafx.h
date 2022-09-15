#pragma once
#include <stdio.h>
#include <thread>
#include <mutex>
#include <future>
#include <iostream>
#include <fstream>
#include <time.h>
#include <vcruntime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define D2R (CV_PI / 180.0)
#define R2D (180.0 / CV_PI)

#define VISION_TOLERANCE 0.0000001
#define MATCH_CANDIDATE_NUM 5
