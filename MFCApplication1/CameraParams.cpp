#include "CameraParams.h"
#include "InverseTransform.h"
#include <iostream>
using namespace std;
vector<Point2f> DstCenterInitial();
vector<Point2f> ImageCentersInitial(Mat wrapSrc);//重心法
vector<Point2f> ImageCentersInitial_Insect(Mat wrapSrc);//交点
vector<Point2f>ImgPoints;
void on_mouse(int event, int x, int y, int flags, void *ustc);

int CameraCalibration()
{
	string src_path = "C:\\Users\\ahaiya\\Documents\\FirstYear\\MachineVision\\SampleImages\\crops002.jpg";
	Mat src = imread(src_path);
	//namedWindow("calibration", CV_WINDOW_AUTOSIZE);
	if (!src.data)return -1;
	CameraParams cameraparams(src);
	cameraparams.GetParams();
	return 0;
}
void CameraParams::GetParams()

{
	ofstream fout("caliberation_result.txt");  /**    保存定标结果的文件     **/

											   /************************************************************************
											   读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化
											   *************************************************************************/
	cout << "开始提取角点………………" << endl;
	int image_count = 9;                    /****    图像数量     ****/
	Size image_size;                         /****     图像的尺寸      ****/
	Size board_size = Size(11, 14);            /****    定标板上每行、列的角点数       ****/
	vector<Point2f> corners;                  /****    缓存每幅图像上检测到的角点       ****/
	vector<vector<Point2f>>  corners_Seq;    /****  保存检测到的所有角点        ****/
	vector<Mat>  image_Seq;

	int count = 0;
	for (int i = 0; i != image_count; i++)
	{
		cout << "Frame #" << i + 1 << "..." << endl;
		string imageFileName;
		std::stringstream StrStm;
		StrStm << i + 1;
		StrStm >> imageFileName;
		imageFileName += ".jpg";
		cv::Mat image = imread("CalibrationImages\\"+imageFileName);
		image_size = image.size();
		//image_size = Size(image.cols , image.rows);  
		/* 提取角点 */
		Mat imageGray;
		cvtColor(image, imageGray, CV_RGB2GRAY);
		bool patternfound = findChessboardCorners(image, board_size, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE +
			CALIB_CB_FAST_CHECK);
		if (!patternfound)
		{
			cout << "can not find chessboard corners!\n";
			waitKey(0);
			exit(1);
		}
		else
		{
			/* 亚像素精确化 */
			cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			count = count + corners.size();
			corners_Seq.push_back(corners);
		}
		image_Seq.push_back(image);
	}
	cout << "角点提取完成！\n";
	/************************************************************************
	摄像机定标
	*************************************************************************/
	cout << "开始定标………………" << endl;
	Size square_size = Size(20, 20);                                      /**** 实际测量得到的定标板上每个棋盘格的大小   ****/
	vector<vector<Point3f>>  object_Points;                                      /****  保存定标板上角点的三维坐标   ****/

	Mat image_points = Mat(1, count, CV_32FC2, Scalar::all(0));          /*****   保存提取的所有角点   *****/
	vector<int>  point_counts;                                          /*****    每幅图像中角点的数量    ****/
	Mat intrinsic_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));                /*****    摄像机内参数矩阵    ****/
	Mat distortion_coeffs = Mat(1, 4, CV_32FC1, Scalar::all(0));            /* 摄像机的4个畸变系数：k1,k2,p1,p2 */
	vector<cv::Mat> rotation_vectors;                                      /* 每幅图像的旋转向量 */
	vector<cv::Mat> translation_vectors;                                  /* 每幅图像的平移向量 */

																		  /* 初始化定标板上角点的三维坐标 */
	for (int t = 0; t<image_count; t++)
	{
		vector<Point3f> tempPointSet;
		for (int i = 0; i<board_size.height; i++)
		{
			for (int j = 0; j<board_size.width; j++)
			{
				/* 假设定标板放在世界坐标系中z=0的平面上 */
				Point3f tempPoint;
				tempPoint.x = i*square_size.width;
				tempPoint.y = j*square_size.height;
				tempPoint.z = 0;
				tempPointSet.push_back(tempPoint);
			}
		}
		object_Points.push_back(tempPointSet);
	}

	/* 初始化每幅图像中的角点数量，这里我们假设每幅图像中都可以看到完整的定标板 */
	for (int i = 0; i< image_count; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}

	/* 开始定标 */
	calibrateCamera(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, 0);
	cout << "定标完成！\n";

	/************************************************************************
	对定标结果进行评价
	*************************************************************************/
	cout << "开始评价定标结果………………" << endl;
	double total_err = 0.0;                   /* 所有图像的平均误差的总和 */
	double err = 0.0;                        /* 每幅图像的平均误差 */
	vector<Point2f>  image_points2;             /****   保存重新计算得到的投影点    ****/

	cout << "每幅图像的定标误差：" << endl;
	cout << "每幅图像的定标误差：" << endl << endl;
	for (int i = 0; i<image_count; i++)
	{
		vector<Point3f> tempPointSet = object_Points[i];
		/****    通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点     ****/
		projectPoints(tempPointSet, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs, image_points2);
		/* 计算新的投影点和旧的投影点之间的误差*/
		vector<Point2f> tempImagePoint = corners_Seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (size_t i = 0; i != tempImagePoint.size(); i++)
		{
			image_points2Mat.at<Vec2f>(0, i) = Vec2f(image_points2[i].x, image_points2[i].y);
			tempImagePointMat.at<Vec2f>(0, i) = Vec2f(tempImagePoint[i].x, tempImagePoint[i].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= point_counts[i];
		cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}
	cout << "总体平均误差：" << total_err / image_count << "像素" << endl;
	fout << "总体平均误差：" << total_err / image_count << "像素" << endl << endl;
	cout << "评价完成！" << endl;

	/************************************************************************
	保存定标结果
	*************************************************************************/
	cout << "开始保存定标结果………………" << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */

	fout << "相机内参数矩阵：" << endl;
	fout << intrinsic_matrix << endl;
	fout << "畸变系数：\n";
	fout << distortion_coeffs << endl;
	for (int i = 0; i<image_count; i++)
	{
		fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		fout << rotation_vectors[i] << endl;

		/* 将旋转向量转换为相对应的旋转矩阵 */
		Rodrigues(rotation_vectors[i], rotation_matrix);
		fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		fout << rotation_matrix << endl;
		fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		fout << translation_vectors[i] << endl;
	}
	cout << "完成保存" << endl;
	fout << endl;

	/************************************************************************
	显示定标结果
	*************************************************************************/
	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);
	cout << "保存矫正图像" << endl;
	for (int i = 0; i != image_count; i++)
	{
		cout << "Frame #" << i + 1 << "..." << endl;
		Mat newCameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
		initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy);
		imwrite("mapx.jpg", mapx);
		imwrite("mapy.jpg", mapy);
		Mat t = image_Seq[i].clone();
		cv::remap(image_Seq[i], t, mapx, mapy, INTER_LINEAR);
		string imageFileName;
		std::stringstream StrStm;
		StrStm << i + 1;
		StrStm >> imageFileName;
		imageFileName += "_d.jpg";
		imwrite("CalibrationImages\\"+imageFileName, t);
	}
	cout << "保存结束" << endl;
	waitKey(0);
}

Mat CameraParams::WrapMatrix()
{
	string srcImagePath = "CalibrationImages\\groud.jpg";
	Mat wrapSrc = imread(srcImagePath);
	vector<Point2f> dstPoints = DstCenterInitial();
	vector<Point2f> srcPoints = ImageCentersInitial_Insect(wrapSrc);
	//Mat Homography = findFundamentalMat(srcPoints, dstPoints, FM_RANSAC, 3, 0.99);//7个及以上点
	Mat Homography = getPerspectiveTransform(srcPoints,dstPoints);
	//
 	Mat dstImg;
	////Rect roi_h = Rect(0, 0, 3, 2);
	////Homography = Homography(roi_h);
	//warpPerspective(wrapSrc, dstImg, Homography,Size(300, 260), CV_INTER_LINEAR);
	//namedWindow("dstImg", CV_WINDOW_KEEPRATIO);
	//imshow("dstImg", dstImg);
	//namedWindow("wrapSrc", CV_WINDOW_KEEPRATIO);
	//imshow("wrapSrc", wrapSrc);
	/*double v;
	for (size_t nrow = 0; nrow < Homography.rows; nrow++)
	{
		for (size_t ncol = 0; ncol < Homography.cols; ncol++)
		{
			v = Homography.at<double>(nrow, ncol);
		}
	}*/
	//double x1 = Center[0].x * Homography.at<double>(0, 0) + Center[0].y * Homography.at<double>(0, 1) + Homography.at<double>(0, 2);
	//double y1 = Center[0].x * Homography.at<double>(1, 0) + Center[0].y * Homography.at<double>(1, 1) + Homography.at<double>(1, 2);
	//vector<Point2d>actual_line;
	//perspectiveTransform(Center, actual_line, Homography);

	return Homography;
	//return wrapSrc;
}
vector<Point2f> DstCenterInitial()
{
	vector<Point2f> Centers;
	//修改+b后乘以k
	int k = 1, b = 40,dy=0;
	//Centers.push_back(Point2f(k*(43.06 + b), k*(dy+185.04)));
	//Centers.push_back(Point2f(k*(22.63 + b), k*(dy+145.90)));
	//Centers.push_back(Point2f(k*(17.96 + b), k*(dy+204.32)));
	//Centers.push_back(Point2f(k*(b + 0), k*(dy-170.49)));
	//Centers.push_back(Point2f(k*(b - 12.55), k*(dy+132.77)));
	//Centers.push_back(Point2f(k*(b - 28.77), k*(dy+220.23)));// 
	//Centers.push_back(Point2f(k*(b - 35.18), k*(dy+157.37)));
	//Centers.push_back(Point2f(k*(b - 36.31), k*(dy + 184.23)));	//
	//Centers.push_back(Point2f(120, 10));	//
	Centers.push_back(Point2f(180,70));	//
	Centers.push_back(Point2f(240, 70));	//
	//Centers.push_back(Point2f(60, 130));	//
	//Centers.push_back(Point2f(120, 130));	//
	Centers.push_back(Point2f(180, 190));	//
	Centers.push_back(Point2f(240, 190));	//
	//Centers.push_back(Point2f(60, 250));	//
	return Centers;
}

vector<Point2f> ImageCentersInitial(Mat wrapSrc)
{
	Mat grayImg;
	cvtColor(wrapSrc, grayImg, COLOR_BGR2GRAY);
	threshold(grayImg, grayImg, 80, 255, 0);
	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;
	findContours(grayImg, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	int i = 0;
	int count = 0;
	Point pt[8];//假设有8个连通区域
	Moments moment;//矩
	vector<Point2f>Center;//创建一个向量保存重心坐标
	for (; i >= 0; i = hierarchy[i][0])//读取每一个轮廓求取重心
	{
		Mat temp(contours.at(i));
		Scalar color(0, 0, 255);
		moment = moments(temp, false);
		if (moment.m00 != 0)//除数不能为0
		{
			pt[i].x = cvRound(moment.m10 / moment.m00);//计算重心横坐标
			pt[i].y = cvRound(moment.m01 / moment.m00);//计算重心纵坐标

		}
		Point p = Point(pt[i].x, pt[i].y);//重心坐标
		//circle(White, p, 1, color, 1, 8);//原图画出重心坐标
		count++;//重心点数或者是连通区域数
		Center.push_back(p);//将重心坐标保存到Center向量中
		//circle(grayImg, p, 3, Scalar(0, 255/(i+1), 0), -1, 8, 0);
	}
	return Center;
}

vector<Point2f> ImageCentersInitial_Insect(Mat wrapSrc)
{
	namedWindow("img", CV_WINDOW_KEEPRATIO);
	while (1) {
		setMouseCallback("img", on_mouse, 0);//调用回调函数
		imshow("img", wrapSrc);
		if (waitKey(100) >= 0)break;
	}
	return ImgPoints;
}
void on_mouse(int event, int x, int y, int flags, void *ustc)
//event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号    
{
	Mat& image = *(cv::Mat*) ustc;
	char temp[16];
	if (event == 1)
	{
		sprintf_s(temp, "(%d,%d)", x, y);
		ImgPoints.push_back(Point2f(x,y));
	}
}