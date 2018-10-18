#pragma once
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <iostream>
using namespace cv;
using namespace std;

typedef struct Result
{
	double angle;
	double offset;
};
class OpencvHelper
{
private:
	
public :
	Mat src_image_;
	Mat tem_image_;
	Mat template_image_;
	Mat homography_;
	Rect ROI_;
	Mat grey_image_;
	Mat thresh_image_;
	//Mat dst_image_;
	//获取图像中导航参数
	Result GetCropRows();
	Vec4i LengthenLine(Vec4i line, Mat draw);

	//获取投影变换后的导航参数
	Result GetResult(vector<Point2f>line);
	vector<Result> results;//每次找出line后存储，与之前结果进行比较~决定取舍。

	//S(x)
	Mat ImgTemplate(Mat Img);
	void GetImage(bool is_from_camera, bool paused);

	//metohods
	vector<Point2f> GetLine_Tradition();
	Result GetLine_Texture();
	vector<Point2f> GetLine();
	int Save(string picpath);

	void SetImage(Mat src);
	void GreyTransform();
	void OTSUBinarize();
	void EXGCalcultate();
	void DilateImage();


};

class AngleResult
{
private:
	unsigned char err_angle_[2];//data6=err_angle_[0],data5=err_angle_[1]
	unsigned char err_offset_[2];//data4=err_offset_[0],data3=err_offset_[1]
	CString strFID = _T("00000081");
	int DectoHex(int dec, unsigned char *hex, int length)
	{
		int i;
		for (i = length - 1; i >= 0; i--)
		{
			hex[i] = (dec % 256) & 0xFF;
			dec /= 256;
		}
		return 0;
	}
	void SetCANData()
	{
		can_data_[0] = 77;
		can_data_[1] = 15;
		can_data_[2] = 10;
		can_data_[3] = err_offset_[1];
		can_data_[4] = err_offset_[0];
		can_data_[5] = err_angle_[1];
		can_data_[6] = err_angle_[0];
		can_data_[7] = 0;
	}
	void SetCANFramID()
	{
		can_frame_ID[0] = (strFID[1] - 48) + (strFID[0] - 48) * 16;
		can_frame_ID[1] = (strFID[3] - 48) + (strFID[2] - 48) * 16;
		can_frame_ID[2] = (strFID[5] - 48) + (strFID[4] - 48) * 16;
		can_frame_ID[3] = (strFID[7] - 48) + (strFID[6] - 48) * 16;
	}
public:
	AngleResult() {};
	AngleResult(double angle,double offset)
	{
		angle_ = angle;
		offset_ = offset;
		UpDate();
	}
	int angle_;
	int offset_;
	unsigned char can_frame_ID[4];
	unsigned char can_data_[8];	
	void UpDate()
	{
		//int a = int;
		DectoHex(angle_ * 128 + 25600, err_angle_, 2);
		DectoHex(offset_ / 5 + 32000, err_offset_, 2);
		SetCANData();
		SetCANFramID();
	}
};

enum OpType {
	OT_ADD,
	OT_SUB,
	OT_MUL,
	OT_DIV,
};

template<class T>
class VecSum {
	OpType type_;
	const T& op1_;
	const T& op2_;
public:
	VecSum(int type, const T& op1, const T& op2) : type_(type), op1_(op1), op2_(op2) {}

	int operator[](const int i) const {
		switch (type_) {
		case OT_ADD: return op1_[i] + op2_[i];
		case OT_SUB: return op1_[i] - op2_[i];
		case OT_MUL: return op1_[i] * op2_[i];
		case OT_DIV: return op1_[i] / op2_[i];
		default: throw "bad type";
		}
	}
};

template<class T>
VecSum<T> operator+(const T& t1, const T& t2) {
	return VecSum<T>(OT_ADD, t1, t2);
}