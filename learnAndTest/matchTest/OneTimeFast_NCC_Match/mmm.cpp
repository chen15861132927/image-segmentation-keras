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
#include "FastNCCMatch.h"
using namespace std;
using namespace cv;
void DrawDashLine(Mat& matDraw, Point ptStart, Point ptEnd, Scalar color1, Scalar color2)
{
	LineIterator itLine(matDraw, ptStart, ptEnd, 8, 0);
	int iCount = itLine.count;
	bool bOdd = false;
	for (int i = 0; i < iCount; i += 1, itLine++)
	{
		if (i % 3 == 0)
		{
			//白色BGR
			(*itLine)[0] = (uchar)color2.val[0];
			(*itLine)[1] = (uchar)color2.val[1];
			(*itLine)[2] = (uchar)color2.val[2];
		}
		else
		{
			//紅色BGR
			(*itLine)[0] = (uchar)color1.val[0];
			(*itLine)[1] = (uchar)color1.val[1];
			(*itLine)[2] = (uchar)color1.val[2];
		}

	}
}

void DrawMarkCross(Mat& matDraw, int iX, int iY, int iLength, Scalar color, int iThickness)
{
	if (matDraw.empty())
		return;
	Point ptC(iX, iY);
	line(matDraw, ptC - Point(iLength, 0), ptC + Point(iLength, 0), color, iThickness);
	line(matDraw, ptC - Point(0, iLength), ptC + Point(0, iLength), color, iThickness);
}

void main(int argc, char** argv)
{
	string templatepath = "C://Dev//GithubOpenCV//GradientDirectionMatch//OneTimeFast_NCC_Match//templeteImage.bmp";
	//string templatepath = "C://Dev//GithubOpenCV//GradientDirectionMatch//Searchimg2//1.bmp";
	FastNCCMatch fastNccMatch;
	fastNccMatch.MaxMatchCount = 5;
	fastNccMatch.angleStart = -180;
	fastNccMatch.angleEnd = 180;
	fastNccMatch.minScore = 0.58;
	fastNccMatch.m_dMaxOverlap = 0.65;

	Mat templateImage = cv::imread(templatepath, IMREAD_UNCHANGED);
	if (templateImage.empty())
	{
		cout << "\nERROR: Could not load Template Image.\n" << templatepath;
		return;
	}
	Mat m_matDst = Mat(templateImage.size(), CV_8U);
	// Convert color image to gray image.
	if (templateImage.channels() == 3)
	{
		cvtColor(templateImage, m_matDst, CV_RGB2GRAY);
	}
	else
	{
		Mat mask = Mat::ones(templateImage.size(), CV_8U);
		copyTo(templateImage, m_matDst, mask);
	}
	//fastNccMatch.LearnPattern(m_matDst);
	//cvNamedWindow("Template", CV_WINDOW_AUTOSIZE);
	//imshow("Template", templateImage);
	//cvWaitKey(0);

	int imageIndex = 1;
	cout << " ------------------------------------\n";
	while (imageIndex <= 10)
	{
		//string searchImagepath = "C://Dev//GithubOpenCV//GradientDirectionMatch//Searchimg2//" + to_string(imageIndex) + ".bmp";
		//string searchImagepath = "C://Dev//GithubOpenCV//GradientDirectionMatch//Searchimg2//5.bmp";
		string searchImagepath = "C://Dev//GithubOpenCV//GradientDirectionMatch//OneTimeFast_NCC_Match//searchImage.bmp";

		cout << "Path:" << searchImagepath << "ms\n";

		//Load Search Image
		Mat  searchImage = cv::imread(searchImagepath, IMREAD_UNCHANGED);
		if (searchImage.empty())
		{
			cout << "\nERROR: Could not load Search Image." << searchImagepath;
			return;
		}

		Mat m_matSrc = Mat(searchImage.size(), IPL_DEPTH_8U, 1);
		// Convert color image to gray image. 
		if (searchImage.channels() == 3)
		{
			cvtColor(searchImage, m_matSrc, CV_RGB2GRAY);
		}
		else
		{
			Mat mask = Mat::ones(m_matSrc.size(), CV_8UC1);
			copyTo(searchImage, m_matSrc, mask);
		}
		Mat matResult;
		fastNccMatch.MatchTemplate(m_matSrc, m_matDst, matResult);

		//const Scalar colorWaterBlue(230, 255, 102);
		//const Scalar colorGoldenrod(15, 185, 255);
		//const Scalar colorGreen(0, 255, 0);

		//for (int resi = 0; resi < outFind.size(); resi++)
		//{
		//	s_SingleTargetMatch item = outFind.at(resi);
		//	DrawDashLine(searchImage, item.ptLT, item.ptLB, colorWaterBlue, colorGoldenrod);
		//	DrawDashLine(searchImage, item.ptLB, item.ptRB, colorWaterBlue, colorGoldenrod);
		//	DrawDashLine(searchImage, item.ptRB, item.ptRT, colorWaterBlue, colorGoldenrod);
		//	DrawDashLine(searchImage, item.ptRT, item.ptLT, colorWaterBlue, colorGoldenrod);
		//	//左上及角落邊框
		//	Point2d ptDis1, ptDis2;
		//	if (templateImage.cols > templateImage.rows)
		//	{
		//		ptDis1 = (item.ptLB - item.ptLT) / 3;
		//		ptDis2 = (item.ptRT - item.ptLT) / 3 * (searchImage.rows / (float)templateImage.cols);
		//	}
		//	else
		//	{
		//		ptDis1 = (item.ptLB - item.ptLT) / 3 * (searchImage.cols / (float)templateImage.rows);
		//		ptDis2 = (item.ptRT - item.ptLT) / 3;
		//	}
		//	DrawDashLine(searchImage, item.ptLT + ptDis1, item.ptLT + ptDis2, colorWaterBlue, colorGoldenrod);
		//	DrawMarkCross(searchImage, item.ptCenter.x, item.ptCenter.y, 5, colorGreen, 1);
		//	cout << "(" << item.ptCenter.x << "," << item.ptCenter.y << "),Angle:" << item.dMatchedAngle << ", Score:" << item.dMatchScore << "  ";
		//}

		//cout << endl << "Total Speed" << ct << "ms" << endl;

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

