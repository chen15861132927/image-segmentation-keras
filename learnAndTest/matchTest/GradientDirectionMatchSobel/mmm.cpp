//***********************************************************************
// Project		    : GeoMatch
// Author           : Shiju P K
// Email			: shijupk@gmail.com
// Created          : 10-01-2010
//
// File Name		: main.cpp
// Last Modified By : Shiju P K
// Last Modified On : 13-07-2010
// Description      : Defines the entry point for the console application.
//
// Copyright        : (c) . All rights reserved.
//***********************************************************************
//
#include "stdafx.h"

using namespace std;
using namespace cv;

float InvSqrt(float x)
{
	float xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i >> 1); // 计算第一个近似根
	x = *(float*)&i;
	x = x * (1.5f - xhalf * x*x); // 牛顿迭代法
	return x;
}

void gradient_cvFastArctan(Mat gx, Mat gy, Mat &gm, Mat &gd,Mat &gmd)
{
	// Calculate gradient magnitude and direction
	for (int i = 0; i < gx.rows; i++)
	{
		//auto _sdx = (short*)(gx.data + gx.step*i);
		//auto _sdy = (short*)(gy.data + gy.step*i);
		for (int j = 0; j < gx.cols; j++)
		{
			auto element_gx = gx.at<short>(i, j);// _sdx[j];
			auto element_gy = gy.at<short>(i, j); //_sdy[j];          // read x, y derivatives
			if (element_gx != 0 || element_gy != 0)
			{
				float gradMag = 1 / InvSqrt((element_gx*element_gx) + (element_gy*element_gy)); //Magnitude = Sqrt(dx^2 +dy^2)

				auto direction = -cvFastArctan((float)element_gy, (float)element_gx);	 //Direction = invtan (Gy / Gx)
				if (direction < -180)
				{
					direction = direction + 360;
				}
				gd.at<short>(i, j) = (short)direction;
				gm.at<short>(i, j) = (short)gradMag;
				gmd.at<Vec2s>(i, j) = Vec2s((short)direction, (short)gradMag);

			}
			else
			{
				gd.at<short>(i, j) = (short)0;
				gm.at<short>(i, j) = (short)0;
				gmd.at<Vec2s>(i, j) = Vec2s(0,0);
			}
		}
	}
}


void main(int argc, char** argv)
{
	string templatepath = "C://Dev//GithubOpenCV//GradientDirectionMatch//Searchimg2//Template1.bmp";
	//string templatepath = "C://Dev//GithubOpenCV//GradientDirectionMatch//Searchimg2//1.bmp";

	Mat templateImage = cv::imread(templatepath, -1);
	if (templateImage.empty())
	{
		cout << "\nERROR: Could not load Template Image.\n" << templatepath;
		return;
	}
	Size tSize = templateImage.size();

	Mat graytImg = Mat(tSize, CV_8U);
	// Convert color image to gray image.
	if (templateImage.channels() == 3)
	{
		cvtColor(templateImage, graytImg, CV_RGB2GRAY);
	}
	else
	{
		Mat mask = Mat::ones(templateImage.cols, templateImage.rows, CV_8U);
		copyTo(templateImage, graytImg, mask);
	}

	//vector<vector<Point>> contours;
	//vector<Vec4i> hierarchy;

	//findContours(graytImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point());
	//Mat imageContours = Mat::zeros(graytImg.size(), CV_8UC1);
	//Mat Contours = Mat::zeros(graytImg.size(), CV_8UC1); //绘制

	//for (int i = 0; i < contours.size(); i++)
	//{
	//	//contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数
	//	for (int j = 0; j < contours[i].size()&& contours[i].size()>1; j++)
	//	{
	//		//绘制出contours向量所有的像素点
	//		Point P = Point(contours[i][j].x, contours[i][j].y);
	//		Contours.at<uchar>(P) = 255;
	//	}

	//	//绘制轮廓
	//	drawContours(imageContours, contours, i, Scalar(255), 1, 8, hierarchy);
	//}
		// 声明输出矩阵
	//cv::Mat src(tSize, CV_8S );
	//cv::GaussianBlur(graytImg, src, cv::Size(3, 3), 3, 3);
	//cv::GaussianBlur(src, src, cv::Size(3, 3), 3, 3);

	Mat grad_x, grad_y, dst;
	Sobel(graytImg, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(graytImg, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);

	Mat gm = Mat(tSize, CV_16S);
	Mat gd = Mat(tSize, CV_16S);
	cv::Mat gmd = cv::Mat(tSize, CV_16SC2, Scalar(0,0));
	gradient_cvFastArctan(grad_x, grad_y, gm, gd, gmd);

	//	-522, -740;
	//合并的
	//addWeighted(grad_x, 0.5, grad_y, 0.5, 0, dst);
	int imageIndex = 1;

	cvNamedWindow("Template", CV_WINDOW_AUTOSIZE);
	//imshow("Template", imageContours);
	cvWaitKey(0);

	cout << " ------------------------------------\n";
	while (imageIndex <= 10)
	{
		string searchImagepath = "C://Dev//GithubOpenCV//GradientDirectionMatch//Searchimg2//" + to_string(imageIndex) + ".bmp";

		cout << "Path:" << searchImagepath << "ms\n";

		//Load Search Image
		Mat  searchImage = cv::imread(searchImagepath, -1);
		if (searchImage.empty())
		{
			cout << "\nERROR: Could not load Search Image." << searchImagepath;
			return;
		}

		CvSize searchSize = cvSize(searchImage.size().width, searchImage.size().height);
		Mat graySearchImg = Mat(searchSize, IPL_DEPTH_8U, 1);
		// Convert color image to gray image. 
		if (searchImage.channels() == 3)
		{
			cvtColor(searchImage, graySearchImg, CV_RGB2GRAY);
		}
		else
		{
			Mat mask = Mat::ones(graySearchImg.cols, graySearchImg.rows, CV_8UC1);
			copyTo(searchImage, graySearchImg, mask);
		}
		//clock_t start_time1 = clock();

		//clock_t finish_time1 = clock();
		//auto total_time = (double)(finish_time1 - start_time1) / CLOCKS_PER_SEC;

		cvNamedWindow("Search Image", CV_WINDOW_AUTOSIZE);
		imshow("Search Image", searchImage);
		cvWaitKey(0);
		imageIndex = imageIndex + 1;
	}

	cout << "\n ------------------------------------\n\n";
	cout << "\n Press any key to exit!";

	cvDestroyWindow("Search Image");
	cvDestroyWindow("Template");
}

