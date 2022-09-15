#include "stdafx.h"
#include "FastNCCMatch.h"
int FastNCCMatch::GetTopLayer(Mat &matTempl, int iMinDstLength)
{
	int iTopLayer = 0;
	int iMinReduceArea = iMinDstLength * iMinDstLength;
	int iArea = matTempl.cols * matTempl.rows;
	while (iArea > iMinReduceArea)
	{
		iArea /= 4;
		iTopLayer++;
	}
	return iTopLayer;
}
void FastNCCMatch::clear()
{
	vector<Mat>().swap(vecPyramid);
	vector<double>().swap(vecDenominatorPartofTemplete);
	vector<double>().swap(vecInvArea);
	vector<double>().swap(vecTempleteMean);
	vector<bool>().swap(vecResultEqual);
}
void FastNCCMatch::resize(int iSize)
{
	vecTempleteMean.resize(iSize);
	vecDenominatorPartofTemplete.resize(iSize, 0);
	vecInvArea.resize(iSize, 1);
	vecResultEqual.resize(iSize, false);
}
void FastNCCMatch::LearnPattern(Mat &templateImage)
{
	totalTime = 0;
	Mat m_matDst = Mat(templateImage.size(), CV_8U);
	// Convert color image to gray image.
	if (templateImage.channels() == 3)
	{
		cvtColor(templateImage, m_matDst, CV_RGB2GRAY);
	}
	else
	{
		Mat mask = Mat::ones(templateImage.cols, templateImage.rows, CV_8U);
		copyTo(templateImage, m_matDst, mask);
	}

	clear();
	int minside = max(m_matDst.rows, m_matDst.cols);
	int iTopLayer = GetTopLayer(m_matDst, (int)sqrt((double)minside));
	buildPyramid(m_matDst, vecPyramid, iTopLayer);
	iBorderColor = mean(m_matDst).val[0] < 128 ? 255 : 0;
	int vecPyramidCount = vecPyramid.size();
	resize(vecPyramidCount);

	for (int i = 0; i < vecPyramidCount; i++)
	{
		Mat currentPyramid = vecPyramid[i];
		double area = (double)currentPyramid.rows * currentPyramid.cols;
		double invArea = 1. / (area);
		Mat templMean, templSdv;//Scalar templMean, templSdv;

		meanStdDev(currentPyramid, templMean, templSdv);
		double templNorm = templSdv.at<double>(0, 0)*templSdv.at<double>(0, 0);// templSdv[0] * templSdv[0] + templSdv[1] * templSdv[1] + templSdv[2] * templSdv[2] + templSdv[3] * templSdv[3];

		if (templNorm < DBL_EPSILON)
		{
			vecResultEqual[i] = true;
		}

		vecInvArea[i] = invArea;
		vecTempleteMean[i] = templMean.at<double>(0, 0);
		vecDenominatorPartofTemplete[i] = templSdv.at<double>(0, 0)*std::sqrt(area);
	}
	bIsPatternLearned = true;
}
Point2f FastNCCMatch::ptRotatePt2f(Point2f ptInput, Point2f ptOrg, double dAngle)
{
	double dWidth = ptOrg.x * 2;
	double dHeight = ptOrg.y * 2;
	double dY1 = dHeight - ptInput.y, dY2 = dHeight - ptOrg.y;

	double dX = (ptInput.x - ptOrg.x) * cos(dAngle) - (dY1 - ptOrg.y) * sin(dAngle) + ptOrg.x;
	double dY = (ptInput.x - ptOrg.x) * sin(dAngle) + (dY1 - ptOrg.y) * cos(dAngle) + dY2;

	dY = -dY + dHeight;
	return Point2f((float)dX, (float)dY);
}
Size FastNCCMatch::GetBestRotationSize(Size sizeSrc, Size sizeDst, double dRAngle)
{
	double dRAngle_radian = dRAngle * D2R;
	Point ptLT(0, 0), ptLB(0, sizeSrc.height - 1), ptRB(sizeSrc.width - 1, sizeSrc.height - 1), ptRT(sizeSrc.width - 1, 0);
	Point2f ptCenter((sizeSrc.width - 1) / 2.0f, (sizeSrc.height - 1) / 2.0f);
	Point2f ptLT_R = ptRotatePt2f(Point2f(ptLT), ptCenter, dRAngle_radian);
	Point2f ptLB_R = ptRotatePt2f(Point2f(ptLB), ptCenter, dRAngle_radian);
	Point2f ptRB_R = ptRotatePt2f(Point2f(ptRB), ptCenter, dRAngle_radian);
	Point2f ptRT_R = ptRotatePt2f(Point2f(ptRT), ptCenter, dRAngle_radian);

	float fTopY = max(max(ptLT_R.y, ptLB_R.y), max(ptRB_R.y, ptRT_R.y));
	float fBottomY = min(min(ptLT_R.y, ptLB_R.y), min(ptRB_R.y, ptRT_R.y));
	float fRightX = max(max(ptLT_R.x, ptLB_R.x), max(ptRB_R.x, ptRT_R.x));
	float fLeftX = min(min(ptLT_R.x, ptLB_R.x), min(ptRB_R.x, ptRT_R.x));

	if (dRAngle > 360)
		dRAngle -= 360;
	else if (dRAngle < 0)
		dRAngle += 360;

	if (fabs(fabs(dRAngle) - 90) < VISION_TOLERANCE || fabs(fabs(dRAngle) - 270) < VISION_TOLERANCE)
	{
		return Size(sizeSrc.height, sizeSrc.width);
	}
	else if (fabs(dRAngle) < VISION_TOLERANCE || fabs(fabs(dRAngle) - 180) < VISION_TOLERANCE)
	{
		return sizeSrc;
	}

	double dAngle = dRAngle;

	if (dAngle > 0 && dAngle < 90)
	{
		;
	}
	else if (dAngle > 90 && dAngle < 180)
	{
		dAngle -= 90;
	}
	else if (dAngle > 180 && dAngle < 270)
	{
		dAngle -= 180;
	}
	else if (dAngle > 270 && dAngle < 360)
	{
		dAngle -= 270;
	}

	float fH1 = sizeDst.width * sin(dAngle * D2R) * cos(dAngle * D2R);
	float fH2 = sizeDst.height * sin(dAngle * D2R) * cos(dAngle * D2R);

	int iHalfHeight = (int)ceil(fTopY - ptCenter.y - fH1);
	int iHalfWidth = (int)ceil(fRightX - ptCenter.x - fH2);

	Size sizeRet(iHalfWidth * 2, iHalfHeight * 2);

	bool bWrongSize = (sizeDst.width < sizeRet.width && sizeDst.height > sizeRet.height)
		|| (sizeDst.width > sizeRet.width && sizeDst.height < sizeRet.height
			|| sizeDst.area() > sizeRet.area());
	if (bWrongSize)
	{
		sizeRet = Size(int(fRightX - fLeftX + 0.5), int(fTopY - fBottomY + 0.5));
	}
	return sizeRet;
}

void FastNCCMatch::CCOEFF_Denominator(cv::Mat& matSrc, cv::Mat& matResult, int iLayer)
{
	if (vecResultEqual[iLayer])
	{
		matResult = Scalar::all(1);
		return;
	}

	Mat currentTemplete = vecPyramid[iLayer];

	Mat sum, sqsum;
	integral(matSrc, sum, sqsum, CV_64F);

	double templeteMean = vecTempleteMean[iLayer];
	double dDenominatorPart_Templete = vecDenominatorPartofTemplete[iLayer];
	double dInvArea = vecInvArea[iLayer];

	for (int i = 0; i < matResult.rows; i++)
	{
		for (int j = 0; j < matResult.cols; j++)
		{
			double p0data = sum.at<double>(i, j);
			double p1data = sum.at<double>(i, j + currentTemplete.cols);
			double p2data = sum.at<double>(i+ currentTemplete.rows, j);
			double p3data = sum.at<double>(i + currentTemplete.rows, j + currentTemplete.cols);

			double q0data = sqsum.at<double>(i, j);
			double q1data = sqsum.at<double>(i, j + currentTemplete.cols);
			double q2data = sqsum.at<double>(i + currentTemplete.rows, j);
			double q3data = sqsum.at<double>(i + currentTemplete.rows, j + currentTemplete.cols);

			double sumFxy = p0data - p1data - p2data + p3data;
			double numerator = matResult.at<float>(i, j) - sumFxy * templeteMean;

			double srcSquare = q0data - q1data - q2data + q3data;
			double denominatorPart_F = srcSquare - (sumFxy * sumFxy *dInvArea);

			denominatorPart_F = MAX(denominatorPart_F, 0);

			double denominator = 0;
			if (denominatorPart_F <= std::min(0.5, 10 * FLT_EPSILON * srcSquare))
			{
				denominator = 0; // avoid rounding errors
			}
			else
			{
				denominator = std::sqrt(denominatorPart_F)*dDenominatorPart_Templete;
			}

			double finalPercent = 0;
			if (fabs(numerator) < denominator)
			{
				finalPercent = numerator / denominator;
			}
			else if (fabs(numerator) < denominator * 1.125)
			{
				finalPercent = numerator > 0 ? 1 : -1;
			}
			matResult.at<float>(i, j) = (float)finalPercent;
		}
	}
}

void FastNCCMatch::CCOEFF_Denominator_1(cv::Mat& matSrc, cv::Mat& matResult, int iLayer)
{
	if (vecResultEqual[iLayer])
	{
		matResult = Scalar::all(1);
		return;
	}

	Mat currentTemplete = vecPyramid[iLayer];

	Mat sum, sqsum;
	integral(matSrc, sum, sqsum, CV_64F);

	double dTemplMean0 = vecTempleteMean[iLayer];
	double dTemplNorm = vecDenominatorPartofTemplete[iLayer];
	double dInvArea = vecInvArea[iLayer];

	for (int i = 0; i < matResult.rows; i++)
	{
		for (int j = 0; j < matResult.cols; j++)
		{
			double p0data = sum.at<double>(i, j);
			double p1data = sum.at<double>(i, j + currentTemplete.cols);
			double p2data = sum.at<double>(i + currentTemplete.rows, j);
			double p3data = sum.at<double>(i + currentTemplete.rows, j + currentTemplete.cols);

			double q0data = sqsum.at<double>(i, j);
			double q1data = sqsum.at<double>(i, j + currentTemplete.cols);
			double q2data = sqsum.at<double>(i + currentTemplete.rows, j);
			double q3data = sqsum.at<double>(i + currentTemplete.rows, j + currentTemplete.cols);

			double num = matResult.at<float>(i, j), sumt;
			double wndMean2 = 0, wndSum2 = 0;
			sumt = p0data - p1data - p2data + p3data;
			wndMean2 += sumt * sumt;
			num -= sumt * dTemplMean0;
			wndMean2 *= dInvArea;

			double sqsumt = q0data - q1data - q2data + q3data;
			wndSum2 += sqsumt;

			double diff2 = MAX(wndSum2 - wndMean2, 0);
			if (diff2 <= std::min(0.5, 10 * FLT_EPSILON * wndSum2))
				sqsumt = 0; // avoid rounding errors
			else
				sqsumt = std::sqrt(diff2)*dTemplNorm;

			if (fabs(num) < sqsumt)
				num /= sqsumt;
			else if (fabs(num) < sqsumt * 1.125)
				num = num > 0 ? 1 : -1;
			else
				num = 0;

			matResult.at<float>(i, j) = (float)num;
		}
	}
}

//From ImageShop
// 4個有符號的32位的數據相加的和。
inline int _mm_hsum_epi32(__m128i V)      // V3 V2 V1 V0
{
	// 實測這個速度要快些，_mm_extract_epi32最慢。
	__m128i T = _mm_add_epi32(V, _mm_srli_si128(V, 8));  // V3+V1   V2+V0  V1  V0  
	T = _mm_add_epi32(T, _mm_srli_si128(T, 4));    // V3+V1+V2+V0  V2+V0+V1 V1+V0 V0 
	return _mm_cvtsi128_si32(T);       // 提取低位 
}
// 基於SSE的字節數據的乘法。
// <param name="Kernel">需要卷積的核矩陣。 </param>
// <param name="Conv">卷積矩陣。 </param>
// <param name="Length">矩陣所有元素的長度。 </param>
inline int IM_Conv_SIMD(unsigned char* pCharKernel, unsigned char *pCharConv, int iLength)
{
	const int iBlockSize = 16, Block = iLength / iBlockSize;
	__m128i SumV = _mm_setzero_si128();
	__m128i Zero = _mm_setzero_si128();
	for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize)
	{
		__m128i SrcK = _mm_loadu_si128((__m128i*)(pCharKernel + Y));
		__m128i SrcC = _mm_loadu_si128((__m128i*)(pCharConv + Y));
		__m128i SrcK_L = _mm_unpacklo_epi8(SrcK, Zero);
		__m128i SrcK_H = _mm_unpackhi_epi8(SrcK, Zero);
		__m128i SrcC_L = _mm_unpacklo_epi8(SrcC, Zero);
		__m128i SrcC_H = _mm_unpackhi_epi8(SrcC, Zero);
		__m128i SumT = _mm_add_epi32(_mm_madd_epi16(SrcK_L, SrcC_L), _mm_madd_epi16(SrcK_H, SrcC_H));
		SumV = _mm_add_epi32(SumV, SumT);
	}
	int Sum = _mm_hsum_epi32(SumV);
	for (int Y = Block * iBlockSize; Y < iLength; Y++)
	{
		Sum += pCharKernel[Y] * pCharConv[Y];
	}
	return Sum;
}
inline bool compareScoreBig2Small(const s_MatchParameter & lhs, const s_MatchParameter & rhs)
{
	return  lhs.dMatchScore > rhs.dMatchScore;
}
inline bool comparePtWithAngle(const pair<Point2f, double> &lhs, const pair<Point2f, double> &rhs)
{
	return lhs.second < rhs.second;
}

//#define ORG
void  FastNCCMatch::MatchTemplate(cv::Mat& matSrc, cv::Mat &matTemplate, cv::Mat& matResult, int iLayer, bool bUseSIMD)
{
	//std::chrono::time_point<std::chrono::high_resolution_clock> t0 = std::chrono::high_resolution_clock::now();
	//cv::imwrite("searchImage.bmp", matSrc);
	//cv::imwrite("templeteImage.bmp", matTemplate);

	if (bUseSIMD)
	{
		//From ImageShop
		matResult = Mat::zeros(matSrc.rows - matTemplate.rows + 1,
			matSrc.cols - matTemplate.cols + 1, CV_32FC1);
		//matResult.create(matSrc.rows - matTemplate.rows + 1,
		//	matSrc.cols - matTemplate.cols + 1, CV_32FC1);
		//matResult.setTo(0);
		//cv::Mat& matTemplate = vecPyramid[iLayer];

		int  t_r_end = matTemplate.rows, t_r = 0;
		for (int r = 0; r < matResult.rows; r++)
		{
			float* r_matResult = matResult.ptr<float>(r);
			uchar* r_source = matSrc.ptr<uchar>(r);
			uchar* r_template, *r_sub_source;
			for (int c = 0; c < matResult.cols; ++c, ++r_matResult, ++r_source)
			{
				r_template = matTemplate.ptr<uchar>();
				r_sub_source = r_source;
				for (t_r = 0; t_r < t_r_end; ++t_r, r_sub_source += matSrc.cols, r_template += matTemplate.cols)
				{
					*r_matResult = *r_matResult + IM_Conv_SIMD(r_template, r_sub_source, matTemplate.cols);
				}
			}
		}
		//From ImageShop
	}
	else
	{
		matchTemplate(matSrc, matTemplate, matResult, CV_TM_CCORR);
	}

	CCOEFF_Denominator(matSrc, matResult, iLayer);
	//std::chrono::time_point<std::chrono::high_resolution_clock> t1 = std::chrono::high_resolution_clock::now();
	//int64_t xxxx = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
	//totalTime = totalTime+ xxxx;
	//cout << "matSrc Row:" << matSrc.rows << " col:" << matSrc.cols << ",matTemplate Row:" << matTemplate.rows << " col:" << matTemplate.cols << ", milliseconds:" << xxxx << ", totalTime:" << totalTime << endl;
}

Point FastNCCMatch::GetNextMaxLoc(Mat & matResult, Point ptMaxLoc, double dMinValue, int iTemplateW, int iTemplateH, double& dMaxValue, double dMaxOverlap)
{
	//比對到的區域完全不重疊 : +-一個樣板寬高
	//int iStartX = ptMaxLoc.x - iTemplateW;
	//int iStartY = ptMaxLoc.y - iTemplateH;
	//int iEndX = ptMaxLoc.x + iTemplateW;

	//int iEndY = ptMaxLoc.y + iTemplateH;
	////塗黑
	//rectangle (matResult, Rect (iStartX, iStartY, 2 * iTemplateW * (1-dMaxOverlap * 2), 2 * iTemplateH * (1-dMaxOverlap * 2)), Scalar (dMinValue), CV_FILLED);
	////得到下一個最大值
	//Point ptNewMaxLoc;
	//minMaxLoc (matResult, 0, &dMaxValue, 0, &ptNewMaxLoc);
	//return ptNewMaxLoc;

	//比對到的區域需考慮重疊比例
	int iStartX = ptMaxLoc.x - iTemplateW * (1 - dMaxOverlap);
	int iStartY = ptMaxLoc.y - iTemplateH * (1 - dMaxOverlap);
	int iEndX = ptMaxLoc.x + iTemplateW * (1 - dMaxOverlap);

	int iEndY = ptMaxLoc.y + iTemplateH * (1 - dMaxOverlap);
	//塗黑
	rectangle(matResult, Rect(iStartX, iStartY, 2 * iTemplateW * (1 - dMaxOverlap), 2 * iTemplateH * (1 - dMaxOverlap)), Scalar(dMinValue), CV_FILLED);
	//得到下一個最大值
	Point ptNewMaxLoc;
	minMaxLoc(matResult, 0, &dMaxValue, 0, &ptNewMaxLoc);
	return ptNewMaxLoc;
}

void FastNCCMatch::GetRotatedROI(Mat& matSrc, Size size, Point2f ptLT, double dAngle, Mat& matROI)
{
	double dAngle_radian = dAngle * D2R;
	Point2f ptC((matSrc.cols - 1) / 2.0f, (matSrc.rows - 1) / 2.0f);
	Point2f ptLT_rotate = ptRotatePt2f(ptLT, ptC, dAngle_radian);
	Size sizePadding(size.width + 6, size.height + 6);


	Mat rMat = getRotationMatrix2D(ptC, dAngle, 1);
	rMat.at<double>(0, 2) -= ptLT_rotate.x - 3;
	rMat.at<double>(1, 2) -= ptLT_rotate.y - 3;
	//平移旋轉矩陣(0, 2) (1, 2)的減，為旋轉後的圖形偏移，-= ptLT_rotate.x - 3 代表旋轉後的圖形往-X方向移動ptLT_rotate.x - 3

	warpAffine(matSrc, matROI, rMat, sizePadding);
}
void FastNCCMatch::FilterWithScore(vector<s_MatchParameter>  &vec, double dScore)
{
	sort(vec.begin(), vec.end(), compareScoreBig2Small);
	int iSize = vec.size(), iIndexDelete = iSize + 1;
	for (int i = 0; i < iSize; i++)
	{
		if ((vec)[i].dMatchScore < dScore)
		{
			iIndexDelete = i;
			break;
		}
	}
	if (iIndexDelete == iSize + 1)//沒有任何元素小於dScore
		return;
	vec.erase(vec.begin() + iIndexDelete, vec.end());
	return;
	//刪除小於比對分數的元素
	vector<s_MatchParameter>::iterator it;
	for (it = vec.begin(); it != vec.end();)
	{
		if (((*it).dMatchScore < dScore))
			it = vec.erase(it);
		else
			++it;
	}
}


void FastNCCMatch::SortPtWithCenter(vector<Point2f>& vecSort)
{
	int iSize = (int)vecSort.size();
	Point2f ptCenter;
	for (int i = 0; i < iSize; i++)
		ptCenter += vecSort[i];
	ptCenter /= iSize;

	Point2f vecX(1, 0);

	vector<pair<Point2f, double>> vecPtAngle(iSize);
	for (int i = 0; i < iSize; i++)
	{
		vecPtAngle[i].first = vecSort[i];//pt
		Point2f vec1(vecSort[i].x - ptCenter.x, vecSort[i].y - ptCenter.y);
		float fNormVec1 = vec1.x * vec1.x + vec1.y * vec1.y;
		float fDot = vec1.x;

		if (vec1.y < 0)//若點在中心的上方
		{
			vecPtAngle[i].second = acos(fDot / fNormVec1) * R2D;
		}
		else if (vec1.y > 0)//下方
		{
			vecPtAngle[i].second = 360 - acos(fDot / fNormVec1) * R2D;
		}
		else//點與中心在相同Y
		{
			if (vec1.x - ptCenter.x > 0)
				vecPtAngle[i].second = 0;
			else
				vecPtAngle[i].second = 180;
		}

	}
	sort(vecPtAngle.begin(), vecPtAngle.end(), comparePtWithAngle);
	for (int i = 0; i < iSize; i++)
		vecSort[i] = vecPtAngle[i].first;
}
void FastNCCMatch::FilterWithRotatedRect(vector<s_MatchParameter> &vec, int iMethod, double dMaxOverLap)
{
	int iMatchSize = (int)vec.size();
	RotatedRect rect1, rect2;
	for (int i = 0; i < iMatchSize - 1; i++)
	{
		if (vec.at(i).bDelete)
			continue;
		for (int j = i + 1; j < iMatchSize; j++)
		{
			if (vec.at(j).bDelete)
				continue;
			rect1 = vec.at(i).rectR;
			rect2 = vec.at(j).rectR;
			vector<Point2f> vecInterSec;
			int iInterSecType = rotatedRectangleIntersection(rect1, rect2, vecInterSec);
			if (iInterSecType == INTERSECT_NONE)//無交集
				continue;
			else if (iInterSecType == INTERSECT_FULL) //一個矩形包覆另一個
			{
				int iDeleteIndex;
				if (iMethod == CV_TM_SQDIFF)
					iDeleteIndex = (vec.at(i).dMatchScore <= vec.at(j).dMatchScore) ? j : i;
				else
					iDeleteIndex = (vec.at(i).dMatchScore >= vec.at(j).dMatchScore) ? j : i;
				vec.at(iDeleteIndex).bDelete = true;
			}
			else//交點 > 0
			{
				double dArea = contourArea(vecInterSec);

				//if (vecInterSec.size () < 5)//一個或兩個交點
				//	continue;
				//else
				{
					int iDeleteIndex;
					//求面積與交疊比例
					SortPtWithCenter(vecInterSec);
					double dRatio = dArea / rect1.size.area();
					//若大於最大交疊比例，選分數高的
					if (dRatio > dMaxOverLap)
					{
						if (iMethod == CV_TM_SQDIFF)
							iDeleteIndex = (vec.at(i).dMatchScore <= vec.at(j).dMatchScore) ? j : i;
						else
							iDeleteIndex = (vec.at(i).dMatchScore >= vec.at(j).dMatchScore) ? j : i;
						vec.at(iDeleteIndex).bDelete = true;
					}
				}
			}
		}
	}
	vector<s_MatchParameter>::iterator it;
	for (it = vec.begin(); it != vec.end();)
	{
		if ((*it).bDelete)
			it = vec.erase(it);
		else
			++it;
	}
}
float FastNCCMatch::Match(Mat &searchImage, vector<s_SingleTargetMatch> &outFind)
{
	if (!bIsPatternLearned)
		return 0;


	CvSize searchSize = cvSize(searchImage.size().width, searchImage.size().height);
	Mat m_matSrc = Mat(searchSize, IPL_DEPTH_8U, 1);
	// Convert color image to gray image. 
	if (searchImage.channels() == 3)
	{
		cvtColor(searchImage, m_matSrc, CV_RGB2GRAY);
	}
	else
	{
		Mat mask = Mat::ones(m_matSrc.cols, m_matSrc.rows, CV_8UC1);
		copyTo(searchImage, m_matSrc, mask);
	}
	std::chrono::time_point<std::chrono::high_resolution_clock> startclick = std::chrono::high_resolution_clock::now();

	//決定金字塔層數 總共為1 + iLayer層
	int iTopLayer = vecPyramid.size() - 1;
	//建立金字塔
	vector<Mat> vecMatSrcPyr;
	buildPyramid(m_matSrc, vecMatSrcPyr, iTopLayer);

	//第一階段以最頂層找出大致角度與ROI
	double dAngleStep = atan(2.0 / max(vecPyramid[iTopLayer].cols, vecPyramid[iTopLayer].rows)) * R2D;

	vector<double> vecAngles;

	for (double dAngle = angleStart; dAngle < angleEnd; dAngle += dAngleStep)
		vecAngles.push_back(dAngle);

	int iTopSrcW = vecMatSrcPyr[iTopLayer].cols;
	int iTopSrcH = vecMatSrcPyr[iTopLayer].rows;
	Point2f ptCenter((iTopSrcW - 1) / 2.0f, (iTopSrcH - 1) / 2.0f);

	int topSeachAngleSize = (int)vecAngles.size();
	vector<s_MatchParameter> vecMatchParameter;
	//Caculate lowest score at every layer
	vector<double> vecLayerScore(iTopLayer + 1, minScore);
	for (int iLayer = 1; iLayer <= iTopLayer; iLayer++)
	{
		vecLayerScore[iLayer] = vecLayerScore[iLayer - 1] * 0.90;
	}

	std::chrono::time_point<std::chrono::high_resolution_clock> searchFirstTime = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < topSeachAngleSize; i++)
	{
		Mat matRotatedSrc, matR = getRotationMatrix2D(ptCenter, vecAngles[i], 1);
		Mat matResult;
		Point ptMaxLoc;
		double dValue, dMaxVal;
		double dRotate = clock();
		Size sizeBest = GetBestRotationSize(vecMatSrcPyr[iTopLayer].size(), vecPyramid[iTopLayer].size(), vecAngles[i]);

		float fTranslationX = (sizeBest.width - 1) / 2.0f - ptCenter.x;
		float fTranslationY = (sizeBest.height - 1) / 2.0f - ptCenter.y;
		matR.at<double>(0, 2) += fTranslationX;
		matR.at<double>(1, 2) += fTranslationY;
		warpAffine(vecMatSrcPyr[iTopLayer], matRotatedSrc, matR, sizeBest, INTER_LINEAR, BORDER_CONSTANT, Scalar(iBorderColor));
		cv::Mat matTemplate = vecPyramid[iTopLayer];
		MatchTemplate(matRotatedSrc, matTemplate, matResult, iTopLayer, true);
		//matchTemplate (matRotatedSrc, vecPyramid[iTopLayer], matResult, CV_TM_CCOEFF_NORMED);

		minMaxLoc(matResult, 0, &dMaxVal, 0, &ptMaxLoc);
		if (dMaxVal < vecLayerScore[iTopLayer])
			continue;
		vecMatchParameter.push_back(s_MatchParameter(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dMaxVal, vecAngles[i]));

		for (int j = 0; j < MaxMatchCount + MATCH_CANDIDATE_NUM - 1; j++)
		{
			ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc, -1, vecPyramid[iTopLayer].cols, vecPyramid[iTopLayer].rows, dValue, m_dMaxOverlap);
			if (dValue < vecLayerScore[iTopLayer])
				continue;
			vecMatchParameter.push_back(s_MatchParameter(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dValue, vecAngles[i]));
		}
	}
	sort(vecMatchParameter.begin(), vecMatchParameter.end(), compareScoreBig2Small);

	//std::chrono::time_point<std::chrono::high_resolution_clock> searchEndTime = std::chrono::high_resolution_clock::now();
	//int64_t firstTime = std::chrono::duration_cast<std::chrono::microseconds> (searchEndTime - searchFirstTime).count();
	//totalTime = totalTime + firstTime;
	//int64_t toplevelTime = std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::high_resolution_clock::now() - startclick).count();

	//cout << "First step Time:" << firstTime << ",toplevelTime:" << toplevelTime << ",totalTime:" << totalTime << endl;
	//record rotated rectangle、ROI and angle
	int iDstW = vecPyramid[iTopLayer].cols;
	int iDstH = vecPyramid[iTopLayer].rows;

	for (int i = 0; i < vecMatchParameter.size(); i++)
	{
		Point2f ptLT, ptRT, ptRB, ptLB;
		double dRAngle = -vecMatchParameter[i].dMatchAngle * D2R;
		ptLT = ptRotatePt2f(vecMatchParameter[i].pt, ptCenter, dRAngle);
		ptRT = Point2f(ptLT.x + iDstW * (float)cos(dRAngle), ptLT.y - iDstW * (float)sin(dRAngle));
		ptLB = Point2f(ptLT.x + iDstH * (float)sin(dRAngle), ptLT.y + iDstH * (float)cos(dRAngle));
		ptRB = Point2f(ptRT.x + iDstH * (float)sin(dRAngle), ptRT.y + iDstH * (float)cos(dRAngle));
		//紀錄旋轉矩形
		Point2f ptRectCenter = Point2f((ptLT.x + ptRT.x + ptLB.x + ptRB.x) / 4.0f, (ptLT.y + ptRT.y + ptLB.y + ptRB.y) / 4.0f);
		vecMatchParameter[i].rectR = RotatedRect(ptRectCenter, vecPyramid[iTopLayer].size(), (float)vecMatchParameter[i].dMatchAngle);
	}

	FilterWithRotatedRect(vecMatchParameter, CV_TM_CCOEFF_NORMED, m_dMaxOverlap);

	int a = 0;
	int iStopLayer = 0;
	//int iSearchSize = min (m_iMaxPos + MATCH_CANDIDATE_NUM, (int)vecMatchParameter.size ());//可能不需要搜尋到全部 太浪費時間
	vector<s_MatchParameter> vecAllResult;
	for (int i = 0; i < (int)vecMatchParameter.size(); i++)
	{
		bool isAddCurrentVMP = false;
		std::chrono::time_point<std::chrono::high_resolution_clock> searcheachSTime = std::chrono::high_resolution_clock::now();
		float beforeAngle = vecMatchParameter[i].dMatchAngle;
		float beforeScore = vecMatchParameter[i].dMatchScore;


		double dRAngle = -vecMatchParameter[i].dMatchAngle * D2R;
		Point2f ptLT = ptRotatePt2f(vecMatchParameter[i].pt, ptCenter, dRAngle);

		double dAngleStep = atan(2.0 / max(iDstW, iDstH)) * R2D;//min改為max
		vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStep;
		vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStep;

		if (iTopLayer <= iStopLayer)
		{
			vecMatchParameter[i].pt = Point2d(ptLT * ((iTopLayer == 0) ? 1 : 2));
			vecAllResult.push_back(vecMatchParameter[i]);
		}
		else
		{
			for (int iLayer = iTopLayer - 1; iLayer >= iStopLayer; iLayer--)
			{
				//搜尋角度
				dAngleStep = atan(2.0 / max(vecPyramid[iLayer].cols, vecPyramid[iLayer].rows)) * R2D;//min改為max
				vector<double> vecAngles;
				//double dAngleS = vecMatchParameter[i].dAngleStart, dAngleE = vecMatchParameter[i].dAngleEnd;
				double dMatchedAngle = vecMatchParameter[i].dMatchAngle;

				for (int i = -1; i <= 1; i++)
				{
					vecAngles.push_back(dMatchedAngle + dAngleStep * i);
				}
				Point2f ptSrcCenter((vecMatSrcPyr[iLayer].cols - 1) / 2.0f, (vecMatSrcPyr[iLayer].rows - 1) / 2.0f);
				int currentLayerAngleSize = (int)vecAngles.size();
				vector<s_MatchParameter> vecNewMatchParameter(currentLayerAngleSize);
				int iMaxScoreIndex = 0;
				double dBigValue = -1;
				for (int j = 0; j < currentLayerAngleSize; j++)
				{
					Mat matResult, matRotatedSrc;
					double dMaxValue = 0;
					Point ptMaxLoc;
					Mat MatSrcPyr = vecMatSrcPyr[iLayer];
					GetRotatedROI(MatSrcPyr, vecPyramid[iLayer].size(), ptLT * 2, vecAngles[j], matRotatedSrc);
					cv::Mat matTemplate = vecPyramid[iLayer];
					MatchTemplate(matRotatedSrc, matTemplate, matResult, iLayer, true);
					//matchTemplate (matRotatedSrc, vecPyramid[iLayer], matResult, CV_TM_CCOEFF_NORMED);
					minMaxLoc(matResult, 0, &dMaxValue, 0, &ptMaxLoc);
					vecNewMatchParameter[j] = s_MatchParameter(ptMaxLoc, dMaxValue, vecAngles[j]);

					if (vecNewMatchParameter[j].dMatchScore > dBigValue)
					{
						iMaxScoreIndex = j;
						dBigValue = vecNewMatchParameter[j].dMatchScore;
					}
				}
				if (vecNewMatchParameter[iMaxScoreIndex].dMatchScore < vecLayerScore[iLayer])
					break;


				double dNewMatchAngle = vecNewMatchParameter[iMaxScoreIndex].dMatchAngle;

				//讓坐標系回到旋轉時(GetRotatedROI)的(0, 0)
				Point2f ptPaddingLT = ptRotatePt2f(ptLT * 2, ptSrcCenter, dNewMatchAngle * D2R) - Point2f(3, 3);
				Point2f pt(vecNewMatchParameter[iMaxScoreIndex].pt.x + ptPaddingLT.x, vecNewMatchParameter[iMaxScoreIndex].pt.y + ptPaddingLT.y);
				//再旋轉
				pt = ptRotatePt2f(pt, ptSrcCenter, -dNewMatchAngle * D2R);

				if (iLayer == iStopLayer)
				{
					vecNewMatchParameter[iMaxScoreIndex].pt = pt * (iStopLayer == 0 ? 1 : 2);
					vecAllResult.push_back(vecNewMatchParameter[iMaxScoreIndex]);
					isAddCurrentVMP = true;
				}
				else
				{
					//更新MatchAngle ptLT
					vecMatchParameter[i].dMatchAngle = dNewMatchAngle;
					vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStep / 2;
					vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStep / 2;
					ptLT = pt;
				}
			}
		}

		std::chrono::time_point<std::chrono::high_resolution_clock> searcheachETime = std::chrono::high_resolution_clock::now();
		int64_t stepTime = std::chrono::duration_cast<std::chrono::microseconds> (searcheachETime - searcheachSTime).count();
		totalTime = totalTime + stepTime;
		//cout << i <<"beforeAngle:"<< beforeAngle <<",beforeScore:"<< beforeScore << ", step Time:" << stepTime  << ",totalTime:" << totalTime ;

		//if (isAddCurrentVMP)
		//{
		//	s_MatchParameter addr = vecAllResult[vecAllResult.size() - 1];
		//	cout <<  "(" << addr.pt.x << "," << addr.pt.y << "),Angle:" << addr.dMatchAngle <<",Score:"<< addr.dMatchScore;
		//}
		//cout<< endl;
	}
	FilterWithScore(vecAllResult, minScore);
	//int64_t steplevelTime = std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::high_resolution_clock::now() - startclick).count();

	//cout << "steplevelTime" << steplevelTime << endl;
	//最後再次濾掉重疊
	iDstW = vecPyramid[iStopLayer].cols, iDstH = vecPyramid[iStopLayer].rows;

	for (int i = 0; i < (int)vecAllResult.size(); i++)
	{
		Point2f ptLT, ptRT, ptRB, ptLB;
		double dRAngle = -vecAllResult[i].dMatchAngle * D2R;
		ptLT = vecAllResult[i].pt;
		ptRT = Point2f(ptLT.x + iDstW * (float)cos(dRAngle), ptLT.y - iDstW * (float)sin(dRAngle));
		ptLB = Point2f(ptLT.x + iDstH * (float)sin(dRAngle), ptLT.y + iDstH * (float)cos(dRAngle));
		ptRB = Point2f(ptRT.x + iDstH * (float)sin(dRAngle), ptRT.y + iDstH * (float)cos(dRAngle));
		//紀錄旋轉矩形
		Point2f ptRectCenter = Point2f((ptLT.x + ptRT.x + ptLB.x + ptRB.x) / 4.0f, (ptLT.y + ptRT.y + ptLB.y + ptRB.y) / 4.0f);
		vecAllResult[i].rectR = RotatedRect(ptRectCenter, vecPyramid[iStopLayer].size(), (float)vecAllResult[i].dMatchAngle);
	}

	FilterWithRotatedRect(vecAllResult, CV_TM_CCOEFF_NORMED, m_dMaxOverlap);
	//最後再次濾掉重疊

	//根據分數排序
	sort(vecAllResult.begin(), vecAllResult.end(), compareScoreBig2Small);
	int64_t executionTime = std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::high_resolution_clock::now() - startclick).count();

	outFind.clear();

	if (vecAllResult.size() != 0)
	{
		int iW = vecPyramid[0].cols, iH = vecPyramid[0].rows;

		for (int i = 0; i < vecAllResult.size(); i++)
		{
			s_SingleTargetMatch sstm;
			double dRAngle = -vecAllResult[i].dMatchAngle * D2R;

			sstm.ptLT = vecAllResult[i].pt;

			sstm.ptRT = Point2d(sstm.ptLT.x + iW * cos(dRAngle), sstm.ptLT.y - iW * sin(dRAngle));
			sstm.ptLB = Point2d(sstm.ptLT.x + iH * sin(dRAngle), sstm.ptLT.y + iH * cos(dRAngle));
			sstm.ptRB = Point2d(sstm.ptRT.x + iH * sin(dRAngle), sstm.ptRT.y + iH * cos(dRAngle));
			sstm.ptCenter = Point2d((sstm.ptLT.x + sstm.ptRT.x + sstm.ptRB.x + sstm.ptLB.x) / 4, (sstm.ptLT.y + sstm.ptRT.y + sstm.ptRB.y + sstm.ptLB.y) / 4);
			sstm.dMatchedAngle = -vecAllResult[i].dMatchAngle;
			sstm.dMatchScore = vecAllResult[i].dMatchScore;

			if (sstm.dMatchedAngle < -180)
				sstm.dMatchedAngle += 360;
			if (sstm.dMatchedAngle > 180)
				sstm.dMatchedAngle -= 360;
			outFind.push_back(sstm);
		}
	}

	return executionTime / 1000.0;
}