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
#define PI acos(-1)

using namespace std;
using namespace cv;

float my_atan5( long y,long x, float * magnitude)
{
	const int angle[] = { 11520, 6801, 3593, 1824, 916, 458, 229, 115, 57, 29, 14, 7, 4, 2, 1 };

	int i = 0;
	int x_new, y_new;
	int angleSum = 0;

	x = x << 10;// 将 X Y 放大一些，结果会更准确   
	y = y << 10;

	for (i = 0; i < 15; i++)
	{
		if (y > 0)
		{
			x_new = x + (y >> i);
			y_new = y - (x >> i);
			x = x_new;
			y = y_new;
			angleSum += angle[i];
		}
		else
		{
			x_new = x - (y >> i);
			y_new = y + (x >> i);
			x = x_new;
			y = y_new;
			angleSum -= angle[i];
		}
		//if (abs(y) < 10)
		//	break;
		//printf("Debug: i = %d angleSum = %d, angle = %d\n", i, angleSum, angle[i]);
	}
	*magnitude = x >> 10;
	return angleSum ;
}


void gradient_decomposition(Mat gx, Mat gy, Mat gm, Mat gd)
{
	int rows = gx.rows;
	int cols = gx.cols;
	// Calculate gradient magnitude and direction
	for (int i = 0; i < rows; i++)
	{
		auto _sdx = (short*)(gx.data + gx.step*i);
		auto _sdy = (short*)(gy.data + gy.step*i);
		auto _gd = (short*)(gd.data + gd.step*i);
		for (int j = 0; j < cols; j++)
		{
			auto element_gx = _sdx[j];
			auto element_gy = _sdy[j];        // read x, y derivatives
			float mag;
			auto direction = my_atan5(element_gy, element_gx, &mag);
			gd.at<uchar>(i, j) = (uchar)direction/256;
			gm.at<uchar>(i, j) = (uchar)mag;

		}
	}
}
float InvSqrt(float x)
{
	float xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i >> 1); // 计算第一个近似根
	x = *(float*)&i;
	x = x * (1.5f - xhalf * x*x); // 牛顿迭代法
	return x;
}

void gradient_cvFastArctan(Mat gx, Mat gy, Mat gm, Mat gd)
{
	int rows = gx.rows;
	int cols = gx.cols;
	// Calculate gradient magnitude and direction
	for (int i = 0; i < rows; i++)
	{
		auto _sdx = (short*)(gx.data + gx.step*i);
		auto _sdy = (short*)(gy.data + gy.step*i);
		auto _gd = (short*)(gd.data + gd.step*i);
		for (int j = 0; j < cols; j++)
		{
			auto element_gx = _sdx[j];
			auto element_gy = _sdy[j];        // read x, y derivatives
			float gradMag = InvSqrt((element_gx*element_gx) + (element_gy*element_gy)); //Magnitude = Sqrt(dx^2 +dy^2)

			auto direction = cvFastArctan((float)element_gy, (float)element_gx);	 //Direction = invtan (Gy / Gx)
			gd.at<uchar>(i, j) = (uchar)direction;	
			gm.at<uchar>(i, j) = (uchar)gradMag;
		}
	}
}

void gradient_normal(Mat gx, Mat gy, Mat gm, Mat gd)
{
	int rows = gx.rows;
	int cols = gx.cols;
	// Calculate gradient magnitude and direction
	for (int i = 0; i < rows; i++)
	{
		auto _sdx = (short*)(gx.data + gx.step*i);
		auto _sdy = (short*)(gy.data + gy.step*i);
		for (int j = 0; j < cols; j++)
		{
			auto element_gx = _sdx[j];
			auto element_gy = _sdy[j];        // read x, y derivatives
			auto direction = atan2((float)element_gy, (float)element_gx);	 //Direction = invtan (Gy / Gx)
			float gradMag = sqrt((element_gx*element_gx) + (element_gy*element_gy)); //Magnitude = Sqrt(dx^2 +dy^2)

			gd.at<uchar>(i, j) = (uchar)direction;	
			gm.at<uchar>(i, j) = (uchar)gradMag;

		}
	}
}

void main(int argc, char** argv)
{
	//string templatepath = "C://Dev//GithubOpenCV//GradientDirectionMatch//Searchimg2//Template.jpg";
	string templatepath = "C://Dev//GithubOpenCV//GradientDirectionMatch//Searchimg2//1.bmp";

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

	Mat grad_x, grad_y, dst;

	Sobel(graytImg, grad_x, CV_16SC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(graytImg, grad_y, CV_16SC1, 0, 1, 3, 1, 0, BORDER_DEFAULT);

	Mat decomposition = Mat(tSize, CV_8U);
	Mat normal = Mat(tSize, CV_8U);
	Mat FastArctan = Mat(tSize, CV_8U);

	Mat gmdecomposition = Mat(tSize, CV_8U);
	Mat gmnormal = Mat(tSize, CV_8U);
	Mat gmFastArctan = Mat(tSize, CV_8U);

	clock_t start_time1 = clock();
	gradient_normal(grad_x, grad_y, normal, gmnormal);
	clock_t finish_time1 = clock();
	auto total1_time = (double)(finish_time1 - start_time1) / CLOCKS_PER_SEC;

	clock_t start_time2 = clock();
	gradient_cvFastArctan(grad_x, grad_y, FastArctan, gmFastArctan);
	clock_t finish_time2 = clock();
	auto total2_time = (double)(finish_time2 - start_time2) / CLOCKS_PER_SEC;

	clock_t start_time3 = clock();
	gradient_decomposition(grad_x, grad_y, decomposition, gmdecomposition);
	clock_t finish_time3 = clock();
	auto total3_time = (double)(finish_time3 - start_time3) / CLOCKS_PER_SEC;

	cout << "gradient_normal:" << total1_time << " gradient_cvFastArctan:" << total2_time << " gradient_decomposition:" << total3_time << "ms\n";

	//	-522, -740;
		//合并的
		//addWeighted(grad_x, 0.5, grad_y, 0.5, 0, dst);
	int imageIndex = 1;

	cvNamedWindow("Template", CV_WINDOW_AUTOSIZE);
	imshow("Template", graytImg);
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

void mainss(int argc, char** argv)
{
	float R = 10;

	int i = 45;
	/*clock_t start_time1 = clock();*/
	for (int i = 0; i <= 20; i++)
	{
		float rad = 1.0*i / 180.0 * PI;
		float y = sin(rad)*R;
		float x = cos(rad)*R;
		auto normal = atan2(y, x) / PI * 180.0;	 //Direction = invtan (Gy / Gx)
		auto cvfastatan = cvFastArctan(y, x);	 //Direction = invtan (Gy / Gx)
		float magnitude;

		auto cordica = my_atan5(y, x, &magnitude)/256.0;	 //Direction = invtan (Gy / Gx)


		cout << "i:" << i << " normal:" << normal << " cvfastatan:" << cvfastatan << " cordica:" << cordica << "\n";

	}

	//clock_t finish_time1 = clock();
	//auto total1_time = (double)(finish_time1 - start_time1) / CLOCKS_PER_SEC;;
	//cout << "atan2:" << total1_time << "ms\n";

	//clock_t start_time2 = clock();
	//for (int i = 0; i <= 360; i++)
	//{
	//	float rad = 1.0*i / 180.0 * PI;
	//	float y = sin(rad)*R;
	//	float x = cos(rad)*R;
	//	auto cvfastatan = cvFastArctan(y, x);	 //Direction = invtan (Gy / Gx)
	//}

	//clock_t finish_time2 = clock();
	//auto total2_time = (double)(finish_time2 - start_time2) / CLOCKS_PER_SEC;;

	//cout << " cvFastArctan:" << total2_time << "ms\n";
	//clock_t start_time3 = clock();
	//for (int i = 0; i <= 360; i++)
	//{
	//	float rad = 1.0*i / 180.0 * PI;
	//	float y = sin(rad)*R;
	//	float x = cos(rad)*R;
	//	float magnitude;
	//	auto cordica = my_atan5(y, x, &magnitude);	 //Direction = invtan (Gy / Gx)
	//}
	//clock_t finish_time3 = clock();
	//auto total3_time = (double)(finish_time3 - start_time3) / CLOCKS_PER_SEC;;

	//cout << " my_atan5:" << total3_time << "ms\n";
}

#define PI acos(-1)

void main3333(int argc, char** argv)
{
	float R = 10;

	int i = 100;
	/*clock_t start_time1 = clock();*/
	//for (int i = 0; i <= 100; i++)
	{
		//float rad = 1.0*i / 180.0 * PI;
		float y = -3;// sin(rad)*R;
		float x = 89;// cos(rad)*R;
		//auto normal = atan2(y, x) / PI * 180.0;	 //Direction = invtan (Gy / Gx)
		auto cvfastatan = cvFastArctan(y, x);	 //Direction = invtan (Gy / Gx)
		//float magnitude;

		//auto cordica = my_atan5(y, x, &magnitude) / 256.0;	 //Direction = invtan (Gy / Gx)


		cout << "i:" << i << " cvfastatan:" << cvfastatan << "\n";

	}
}