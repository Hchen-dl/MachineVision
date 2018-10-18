#include "stdafx.h"
#include <math.h>
#include<numeric>
#define PI 3.14159;
#include "OpencvHelper.h"

using namespace std;
Result OpencvHelper::GetCropRows()
{
	vector<Point2f> cropline;
	Result result;
	result.angle = 0;
	result.offset = 0;
	//int spatialRad = 7, colorRad = 7, maxPryLevel = 1;
	//
	//pyrMeanShiftFiltering(src_image_, src_image_, spatialRad, colorRad, maxPryLevel);
	EXGCalcultate();
	//GreyTransform();
	OTSUBinarize();
	
	cropline=GetLine_Tradition();
	//result=GetLine_Texture();
	/*for (int r = 1; r<thresh_image_.rows; r = r + 10)
	{
		for (int c = 1; c < thresh_image_.cols; c++)
		{
			int x = thresh_image_.data[r, c];
			x;
		}
	}*/
	//
	//namedWindow("thresh_", CV_WINDOW_KEEPRATIO);
	//imshow("thresh_", thresh_image_);
	//DilateImage();
	//cropline = GetLine();
	result = GetResult(cropline);
	return result;
}
void OpencvHelper::SetImage(Mat src)
{
	if(!src.empty())
	src_image_ = src;
};
Mat OpencvHelper::ImgTemplate(Mat Img)
{
	//	
	Img.convertTo(Img, CV_32FC1);
	Img = Img / 255;//以免太大
	Mat mTemp(1, Img.cols, CV_32FC1, Scalar(0));
	reduce(Img, mTemp, 1, CV_REDUCE_SUM);
	return mTemp;
}
void OpencvHelper::GetImage(bool is_from_camera, bool paused)
{
	{
		VideoCapture cap;
		//cap.open(0);//0 for computer while 1 for USB camera.
		cap.open("C:\\Users\\ahaiya\\Documents\\FirstYear\\MachineVision\\SampleImages\\CH4.avi");
		if (!cap.isOpened())
		{
			cout << "不能初始化摄像头\n";
			waitKey(0);
		}
		//  //方法2  
		while (is_from_camera)
		{

			if (!paused)
			{
				cap >> tem_image_;				
				if (tem_image_.empty())
					break;
				else
				{
					ROI_ = Rect(tem_image_.rows / 2, tem_image_.cols / 4,  tem_image_.cols / 2, tem_image_.rows / 2);
					src_image_ = tem_image_(ROI_);
				}
				GetCropRows();

				cv::imshow("view", src_image_);    //显示当前帧  
				waitKey(20);  //延时30ms  
			}
			else break;
		}
	}
}
vector<Point2f> OpencvHelper::GetLine_Tradition()
{
	vector<Point2f> crop_line;//直线
	vector<cv::Vec4f> lines;
	vector<Point2d> line_center;//直线中心，x为left，y为right。
	int left, right;
	int crop_width = 10;
	Mat points = Mat::zeros(thresh_image_.rows, thresh_image_.cols, CV_8UC1);
	//求取中点
	for (int r=0;r<thresh_image_.rows;r=r+1)
	{
		left = 0, right = 0;
		for (int c = 1; c < thresh_image_.cols-1; c++)
		{
			if (thresh_image_.at<uchar>(r, c-1) == 0 && thresh_image_.at<uchar>(r, c) > 0)
				right=left = c;
			if (thresh_image_.at<uchar>(r, c) > 0 && thresh_image_.at<uchar>(r, c+1) == 0)
				right = c+1;
			if (right - left > crop_width)
				//points.at<uchar>(r, (left + right) / 2) = 255;
			points.at<uchar>(r, (left + right) / 2) = 255;
		}
	}
	namedWindow("points", CV_WINDOW_KEEPRATIO);
	imshow("points", points);
	cv::HoughLinesP(points, lines, 1, 3.14159/180, 30,30,30);//HoughLines的lines是vector2f.参数6为threshold.

	int deta=points.cols;
	int id=0;
	for (size_t i = 0; i < lines.size(); i++)
	{
		//float rho = lines[i][0], theta = lines[i][1];
		//Point pt1, pt2;
		//double a = cos(theta), b = sin(theta);
		//double x0 = a * rho, y0 = b * rho;
		//pt1.x = cvRound(x0 + 1000 * (-b));
		//pt1.y = cvRound(y0 + 1000 * (a));
		//pt2.x = cvRound(x0 - 1000 * (-b));
		//pt2.y = cvRound(y0 - 1000 * (a));
		//line(grey_image_, pt1, pt2, Scalar(55, 100, 195), 1, CV_AA);	

		//求取最中央线段
		Vec4i l = lines[i];//Vec4i 就是Vec<int, 4>，里面存放４个int
		if (abs(l[0] + l[2] - points.cols) < deta)
		{
			deta = abs(l[0] + l[2] - points.cols);
			id = i;
		}
		//line(src_image_, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
	}
	if (lines.size() > 0)
	{

		crop_line.push_back(Point2f(lines[id][0], lines[id][1]));
		crop_line.push_back(Point2f(lines[id][2], lines[id][3]));

		line(src_image_, Point(lines[id][0], lines[id][1]), Point(lines[id][2], lines[id][3]), Scalar(0, 0, 255), 3, CV_AA);
		//return lines[id];
	}
	else
	{
		crop_line.push_back(Point2f(0, 0));
		crop_line.push_back(Point2f(0, 10));
	}
	//namedWindow("src", CV_WINDOW_KEEPRATIO);
	//imshow("src", src_image_);
	return crop_line;
}
Result OpencvHelper::GetLine_Texture()
{
	Result result;

	Mat overHeadImg;
	warpPerspective(grey_image_, overHeadImg, homography_, Size(  grey_image_.rows, grey_image_.cols));
	namedWindow("overHeadImg", CV_WINDOW_KEEPRATIO);
	imshow("overHeadImg", overHeadImg);
	
	vector<Point2f> OriginCorners;
	OriginCorners.push_back(Point(0,0));
	OriginCorners.push_back(Point(overHeadImg.rows,0));
	OriginCorners.push_back(Point(0,overHeadImg.cols));
	OriginCorners.push_back(Point(overHeadImg.rows, overHeadImg.cols));
	Mat originTemp = ImgTemplate(overHeadImg);
	//skew the image
	Mat skewImg;
	double geatestVariance=0;
	double meanVariance = 0;
	int greatestID=0;
	for (int theta = -30; theta < 30; theta++)
	{
		vector<Point2f> DstCorners;
		DstCorners.push_back(Point2f(0, -0.5*overHeadImg.rows*tan(theta / 180)));
		DstCorners.push_back(Point2f(overHeadImg.rows, 0.5*overHeadImg.rows*tan(theta / 180)));
		DstCorners.push_back(Point2f(0, -0.5*overHeadImg.rows*tan(theta / 180) + overHeadImg.cols));
		DstCorners.push_back(Point2f(overHeadImg.rows, 0.5*overHeadImg.rows*tan(theta /180)+overHeadImg.cols));
		Mat skewMat= getPerspectiveTransform(OriginCorners, DstCorners);
		warpPerspective(overHeadImg, skewImg, skewMat, overHeadImg.size());
		//namedWindow("skewImg", CV_WINDOW_KEEPRATIO);
		//imshow("skewImg", skewImg);
		//waitKey(500);

		//calculate S(x);
		Mat skewTemp = ImgTemplate(skewImg);
		Mat sumTemp= originTemp+ skewTemp;
		//transform(originTemp.begin(), originTemp.end(), skewTemp.begin(), sumTemp.begin(), plus<int>());
		//calculate variance
		Scalar mean;  //均值
		Scalar stddev;  //标准差
		cv::meanStdDev(sumTemp, mean, stddev);  //计算均值和标准差
		double mean_pxl = mean.val[0];
		double stddev_pxl = stddev.val[0];
		//choose greatest variance
		if (stddev_pxl > geatestVariance)
		{
			greatestID = theta;
			geatestVariance = stddev_pxl;
		}
		meanVariance += stddev_pxl ;
	}
	meanVariance = meanVariance / 61;
	result.angle = greatestID;
	//get offset;
	Mat matchMat;
	if (!template_image_.data)
	{
		template_image_ = grey_image_(ROI_);
	}
	matchTemplate(grey_image_, template_image_, matchMat, CV_TM_CCORR_NORMED);
	normalize(matchMat, matchMat, 0, 1, NORM_MINMAX, -1, Mat());
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;
	minMaxLoc(matchMat, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	result.offset= (double)(maxLoc.x-ROI_.x);//
	//a threshold;
	double threshVariance = 1.1;
	if (geatestVariance/meanVariance>threshVariance)
		template_image_ = grey_image_(ROI_);
	//
	//namedWindow("template_image_", CV_WINDOW_KEEPRATIO);
	//imshow("template_image_", template_image_);
	return result;
}

vector<Point2f> OpencvHelper::GetLine()
{
	vector<Point2f> crop_line;
	int crop_circle = 100;
	RNG rng(12345);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// 寻找轮廓
	findContours(thresh_image_, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// 对每个找到的轮廓创建可倾斜的边界框和椭圆
	vector<RotatedRect> minRect(contours.size());
	vector<RotatedRect> minEllipse(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		minRect[i] = minAreaRect(Mat(contours[i]));
		if (contours[i].size() > 10)
		{
			minEllipse[i] = fitEllipse(Mat(contours[i]));
		}
	}

	/// 绘出轮廓及其可倾斜的边界框和边界椭圆
	Mat test = Mat::zeros(thresh_image_.size(), CV_8UC3);
	Mat drawing = src_image_;
	Scalar color;
	Point2d pc, p1, p2, pcenter;
	double line_angle;
	double max_line_angle = 90; int max_id = 0;
	double line_result[3];
	for (int i = 0; i< contours.size(); i++)
	{
		if (arcLength(contours[i], 1) >crop_circle)
		{
			color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			// contour
			//drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
			// ellipse
			//ellipse(drawing, minEllipse[i], color, 2, 8);
			//line
			pc = minEllipse[i].center;
			line_angle = (minEllipse[i].angle + 90) / 180 * 3.1415;
			p1.x = pc.x + minEllipse[i].size.height*0.5*cos(line_angle);
			p1.y = pc.y + minEllipse[i].size.height*0.5*sin(line_angle);
			p2.x = pc.x - minEllipse[i].size.height*0.5*cos(line_angle);
			p2.y = pc.y - minEllipse[i].size.height*0.5*sin(line_angle);
			//line(drawing, p1, p2, color, 5);
			//剔除其他作物行获取一条作物行
			if (abs(max_line_angle) > abs(minEllipse[i].angle))
			{
				max_line_angle = minEllipse[i].angle;
				pcenter = pc;
				max_id = i;
			}

			// rotated rectangle
			/*Point2f rect_points[4];
			minRect[i].points(rect_points);
			for (int j = 0; j < 4; j++)
				line(grey_image_, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);*/
		}
	}
	//namedWindow("thresh", CV_WINDOW_KEEPRATIO);
	//imshow("thresh", thresh_image_);
	//保存/显示最后的作物行
	line_result[0] = pcenter.x;
	line_result[1] = pcenter.y;
	line_result[2] = max_line_angle + 90;
	max_line_angle = (max_line_angle + 90) / 180 * 3.1415;
	p1.x = pcenter.x + minEllipse[max_id].size.height*0.5*cos(max_line_angle);
	p1.y = pcenter.y + minEllipse[max_id].size.height*0.5*sin(max_line_angle);
	p2.x = pcenter.x - minEllipse[max_id].size.height*0.5*cos(max_line_angle);
	p2.y = pcenter.y - minEllipse[max_id].size.height*0.5*sin(max_line_angle);

	color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	//line(drawing, p1, p2, color, 5);
	
	/// 结果在窗体中显示
	//namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	//imshow("view", drawing);
	src_image_ = drawing;
	crop_line.push_back(p1);
	crop_line.push_back(p2);
	return crop_line;
	
	//return drawing;
}

int OpencvHelper::Save(string picpath)
{
	if (src_image_.data)
	{
		imwrite(picpath, src_image_);
		return 1;
	}
	else return 0;
}
void OpencvHelper::EXGCalcultate()
{
	//Mat exgImage;
	double coefficient[2][3] = { { -1,1.8,-1 },{ 0.5,0.3,0.2 } };//存储EXG系数
	std::vector<Mat> channels;
	//Mat aChannels[3];
	//src为要分离的Mat对象  
	//split(src_image_, aChannels);              //利用数组分离  
	split(src_image_, channels);             //利用vector对象分离  
	for (int i = 0; i < 3; i++)
	{
		channels[i].convertTo(channels[i], CV_32F);//计算时使用32F格式，计算完成之后转化成8U；			
		channels[i] = channels[i] * coefficient[0][i];
	}
	grey_image_.convertTo(grey_image_, CV_32F);
	grey_image_ = channels[0] + channels[1] + channels[2];
	grey_image_.convertTo(grey_image_, CV_8UC1);
	ROI_ = Rect(30, 30, grey_image_.cols -30, grey_image_.rows-30);
	//namedWindow("exg", CV_WINDOW_KEEPRATIO);
	//imshow("exg", grey_image_);
	//("exg", exgImage.cols / 6, exgImage.rows / 6);
}
void OpencvHelper::GreyTransform()
{
	cv::cvtColor(src_image_, grey_image_, CV_BGR2GRAY);
	int channels = src_image_.channels();
	int n_rows = src_image_.rows;
	int n_cols = src_image_.cols;
	if (src_image_.isContinuous())
	{
		n_cols *= n_rows;
		n_rows = 1;
	}
	int i, j, tem;
	// size
	int L = 10;
	uchar* p;
	uchar* q;
	uchar* r;
	for (i = 0; i < n_rows; ++i)
	{
		p = src_image_.ptr<uchar>(i);
		q = grey_image_.ptr<uchar>(i);
		for (j = 0; j < n_cols; ++j)
		{
			tem = p[j * 3 - 1] * 1.8 - p[j * 3] - p[j * 3 - 2];
			if (tem <= 0)
				q[j] = 0;
			else
			{
				if (tem> 255)
					q[j] = 255;
				else q[j] = tem;
			}
		}
	}
}
void OpencvHelper::OTSUBinarize()
{
	threshold(grey_image_, thresh_image_, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	
}
Result OpencvHelper::GetResult(vector<Point2f>line)
{
	//float matrix[3][3] = { { 1, 0, 1},{ 0, 1, 0},{ 2, 1, 2 } };
	///Mat H(Size(3, 3), CV_32F, matrix);//homography matrix

	Result result;
	double distance0 = src_image_.cols/2;
	double heading_angle = 90;
	vector<Point2f> actual_line;
	actual_line = line;
	//actual_line = line;//
	//vector<Point2f> test;
	//test.push_back(Point2f(0, src_image_.cols/2));
	//test.push_back(Point2f(220, src_image_.cols / 2));
	//perspectiveTransform(line, actual_line, homography_);
	//perspectiveTransform(test, actual_line, homography_);
	//double x1 = line[0].x * homography_.at<double>(0, 0) + line[0].y * homography_.at<double>(0, 1) + homography_.at<double>(0, 2);
	//double y1 = line[0].x * homography_.at<double>(1, 0) + line[0].y * homography_.at<double>(1, 1) + homography_.at<double>(1, 2);
	//double x2 = line[0].x * homography_.at<double>(0, 0) + line[1].y * homography_.at<double>(0, 1) + homography_.at<double>(0, 2);
	//double y2 = line[1].x * homography_.at<double>(1, 0) + line[1].y * homography_.at<double>(1, 1) + homography_.at<double>(1, 2);
	//actual_line.push_back(Point2f(x1, y1));
	//actual_line.push_back(Point2f(x2, y2));
	if ((actual_line[0].x - actual_line[1].x) == 0)
		{
			result.angle = 0;
			result.offset = actual_line[0].y - distance0;
		}
	else
		{
			result.angle = 180*(atan((actual_line[0].y- actual_line[1].y)/ (actual_line[0].x - actual_line[1].x)))/PI;
			if (result.angle < 0)
			{
				result.angle = -(result.angle + 90);
			}
			else result.angle = 90- result.angle;
			result.offset =(actual_line[0].x+(src_image_.rows-actual_line[0].y)*(actual_line[0].x-actual_line[1].x)/(actual_line[0].y-actual_line[1].y)-distance0)/10;
			//result.offset =(actual_line[0].y+(src_image_.rows-actual_line[0].x)*(actual_line[0].y-actual_line[1].y)/(actual_line[0].x-actual_line[1].x)-distance0)/10;
		}
	//if ()
	//{

	//}
	return result;
};

Vec4i OpencvHelper::LengthenLine(Vec4i line, Mat draw)
{
	cv::Vec4i v = line;
	if (abs(v[2] - v[0]) >abs(v[3] - v[1]))
	{
		line[0] = 0;
		line[1] = ((float)v[1] - v[3]) / (v[0] - v[2]) * -v[0] + v[1];
		line[2] = draw.cols;
		line[3] = ((float)v[1] - v[3]) / (v[0] - v[2]) * (draw.cols - v[2]) + v[3];
	}
	else
	{
		line[0] = ((float)v[0] - v[2]) / (v[1] - v[3]) * -v[1] + v[0];
		line[1] = 0;
		line[2] = ((float)v[0] - v[2]) / (v[1] - v[3])*(draw.rows - v[1]) + v[0];
		line[3] = draw.rows;
	}
	return v;
}
void OpencvHelper::DilateImage()
{
	int dilation_type = MORPH_CROSS;
	int dilation_elem = 3;
	int dilation_size = 5;
	Mat element = getStructuringElement(dilation_type,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	//腐蚀操作
	erode(thresh_image_, thresh_image_, element);
	// 膨胀
	dilate(thresh_image_, thresh_image_, element);

}