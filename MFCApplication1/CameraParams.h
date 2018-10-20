#pragma once
#ifndef TESTOPENCV170908_LINEDETECT_XUEJINLIN_CAMERAPARAMS_H_
#define TESTOPENCV170908_LINEDETECT_XUEJINLIN_CAMERAPARAMS_H_
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <stdlib.h>
#include <stdio.h>

using namespace cv;
class CameraParams
{
private:
	Mat input_image_;
public:
	CameraParams(Mat input_image)
	{
		input_image_ = input_image;
	}
	void GetParams();
	Mat WrapMatrix();
};
#endif // !TESTOPENCV170908_LINEDETECT_XUEJINLIN_CAMERAPARAMS_H_
