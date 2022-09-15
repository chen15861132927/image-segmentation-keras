#pragma once

#include "stdafx.h"

struct s_TemplData
{
	vector<Mat> vecPyramid;
	vector<double> vecTemplMean;
	vector<double> vecTemplNorm;
	vector<double> vecInvArea;
	vector<bool> vecResultEqual1;
	bool bIsPatternLearned;
	int iBorderColor;
	void clear()
	{
		vector<Mat>().swap(vecPyramid);
		vector<double>().swap(vecTemplNorm);
		vector<double>().swap(vecInvArea);
		vector<double>().swap(vecTemplMean);
		vector<bool>().swap(vecResultEqual1);
	}
	void resize(int iSize)
	{
		vecTemplMean.resize(iSize);
		vecTemplNorm.resize(iSize, 0);
		vecInvArea.resize(iSize, 1);
		vecResultEqual1.resize(iSize, false);
	}
	s_TemplData()
	{
		bIsPatternLearned = false;
	}
};
struct s_MatchParameter
{
	Point2d pt;
	double dMatchScore;
	double dMatchAngle;
	//Mat matRotatedSrc;
	Rect rectRoi;
	double dAngleStart;
	double dAngleEnd;
	RotatedRect rectR;
	Rect rectBounding;
	bool bDelete;

	//double vecResult[3][3];//for subpixel
	int iMaxScoreIndex;//for subpixel
	bool bPosOnBorder;
	Point2d ptSubPixel;
	double dNewAngle;

	s_MatchParameter(Point2f ptMinMax, double dScore, double dAngle)//, Mat matRotatedSrc = Mat ())
	{
		pt = ptMinMax;
		dMatchScore = dScore;
		dMatchAngle = dAngle;

		bDelete = false;
		dNewAngle = 0.0;

		bPosOnBorder = false;
	}
	s_MatchParameter()
	{
		double dMatchScore = 0;
		double dMatchAngle = 0;
	}
	~s_MatchParameter()
	{

	}
 
};
struct s_SingleTargetMatch
{
	Point2d ptLT, ptRT, ptRB, ptLB, ptCenter;
	double dMatchedAngle;
	double dMatchScore;
};

class FastNCCMatch
{
private:
	s_TemplData m_TemplData;
	int m_iMinReduceArea = 64;
	int64_t totalTime;

	void SortPtWithCenter(vector<Point2f>& vecSort);

	int GetTopLayer(Mat &matTempl, int iMinDstLength);
	Size GetBestRotationSize(Size sizeSrc, Size sizeDst, double dRAngle);
	inline Point2f ptRotatePt2f(Point2f ptInput, Point2f ptOrg, double dAngle);
	void CCOEFF_Denominator(cv::Mat& matSrc, cv::Mat &matTemplate, cv::Mat& matResult);
	Point GetNextMaxLoc(Mat & matResult, Point ptMaxLoc, double dMinValue, int iTemplateW, int iTemplateH, double& dMaxValue, double dMaxOverlap);
	void GetRotatedROI(Mat& matSrc, Size size, Point2f ptLT, double dAngle, Mat& matROI);
	void FilterWithScore(vector<s_MatchParameter> &vec, double dScore);
	void FilterWithRotatedRect(vector<s_MatchParameter>  &vec, int iMethod, double dMaxOverLap);

public:
	int MaxMatchCount=5;

	int angleStart = -180;
	int angleEnd = 180;
	float minScore = 0.6;
	float m_dMaxOverlap = 0.75;
	void LearnPattern(Mat &m_matDst);
	float Match(Mat &m_matSrc, vector<s_SingleTargetMatch> &outFind);
	void  MatchTemplate(cv::Mat& matSrc, cv::Mat &matTemplate, cv::Mat& matResult);

};

