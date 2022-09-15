#pragma once

#include "stdafx.h"

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
	vector<Mat> vecPyramid;
	vector<double> vecTempleteMean;
	vector<double> vecDenominatorPartofTemplete;
	vector<double> vecInvArea;
	vector<bool> vecResultEqual;
	bool bIsPatternLearned;
	int iBorderColor;
	int m_iMinReduceArea = 64;
	int64_t totalTime;

	void SortPtWithCenter(vector<Point2f>& vecSort);

	int GetTopLayer(Mat &matTempl, int iMinDstLength);
	Size GetBestRotationSize(Size sizeSrc, Size sizeDst, double dRAngle);
	inline Point2f ptRotatePt2f(Point2f ptInput, Point2f ptOrg, double dAngle);
	void CCOEFF_Denominator(cv::Mat& matSrc, cv::Mat& matResult, int iLayer);
	void CCOEFF_Denominator_1(cv::Mat& matSrc, cv::Mat& matResult, int iLayer);

	Point GetNextMaxLoc(Mat & matResult, Point ptMaxLoc, double dMinValue, int iTemplateW, int iTemplateH, double& dMaxValue, double dMaxOverlap);
	void GetRotatedROI(Mat& matSrc, Size size, Point2f ptLT, double dAngle, Mat& matROI);
	void FilterWithScore(vector<s_MatchParameter> &vec, double dScore);
	void FilterWithRotatedRect(vector<s_MatchParameter>  &vec, int iMethod, double dMaxOverLap);
	void clear();
	void resize(int iSize);
public:
	int MaxMatchCount=5;

	int angleStart = -180;
	int angleEnd = 180;
	float minScore = 0.6;
	float m_dMaxOverlap = 0.75;
	void LearnPattern(Mat &m_matDst);
	float Match(Mat &m_matSrc, vector<s_SingleTargetMatch> &outFind);
	void MatchTemplate(cv::Mat& matSrc, cv::Mat &matTemplate, cv::Mat& matResult, int iLayer, bool bUseSIMD);

};

