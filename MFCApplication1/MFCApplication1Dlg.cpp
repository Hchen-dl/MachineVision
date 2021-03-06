
// MFCApplication1Dlg.cpp: 实现文件
//

#include "stdafx.h"
#include "MainAPP.h"
#include "ControlCAN.h"
#pragma comment(lib,"controlcan.lib")
#include "MFCApplication1Dlg.h"
#include "afxdialogex.h"
#include "OpencvHelper.h"
#include "CameraParams.h"
#ifdef _DEBUG
#define new DEBUG_NEW
#endif
#define ID_TIMER_SHOWCNT 1
#define ID_TIMER_SHOWCNT_TM 1000

OpencvHelper opencv_helper;
//Mat Homography(Size(3, 3), CV_64F);
Mat Homography;
bool paused=false;
void HomoInitial(int OperationFlag);//0写入1读取
// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CMFCApplication1Dlg 对话框



CMFCApplication1Dlg::CMFCApplication1Dlg(CWnd* pParent /*=NULL*/)
	: CDialog(IDD_MAIN_DIALOG, pParent)
	, m_status(_T(""))
	, m_is_from_camera(FALSE)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	m_connect = 0;
	m_devtype = 0;
	m_devind = 0;
}

void CMFCApplication1Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//  DDX_Control(pDX, IDC_PIC_BOX, m_pic_control);
	DDX_Text(pDX, IDC_SHOWSTATUS, m_status);
	DDX_Check(pDX, IDC_IS_FROM_CAMERA, m_is_from_camera);
}

BEGIN_MESSAGE_MAP(CMFCApplication1Dlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDOK, &CMFCApplication1Dlg::OnBnClickedOk)
	ON_BN_CLICKED(IDC_BTNCHOOSE, &CMFCApplication1Dlg::OnBnClickedBtnchoose)
	ON_BN_CLICKED(IDC_IS_FROM_CAMERA, &CMFCApplication1Dlg::OnBnClickedIsFromCamera)
	ON_BN_CLICKED(IDC_BTNPause, &CMFCApplication1Dlg::OnBnClickedBtnpause)
	ON_BN_CLICKED(IDC_PROCEED, &CMFCApplication1Dlg::OnBnClickedProceed)
	ON_BN_CLICKED(IDC_BTNSAVE, &CMFCApplication1Dlg::OnBnClickedBtnsave)
	ON_BN_CLICKED(IDCANCEL, &CMFCApplication1Dlg::OnBnClickedCancel)
	ON_BN_CLICKED(ID_QUIT, &CMFCApplication1Dlg::OnBnClickedQuit)
	ON_BN_CLICKED(IDC_BTNCAN, &CMFCApplication1Dlg::OnBnClickedBtncan)
	ON_BN_CLICKED(IDC_BTN_STARTCAN, &CMFCApplication1Dlg::OnBnClickedBtnStartcan)
	ON_BN_CLICKED(IDC_BTN_CONECTCAN, &CMFCApplication1Dlg::OnBnClickedBtnConectcan)
	ON_BN_CLICKED(IDC_BTN_SEND, &CMFCApplication1Dlg::OnBnClickedBtnSend)
	ON_BN_CLICKED(IDC_BTN_CALIBRATE, &CMFCApplication1Dlg::OnBnClickedBtnCalibrate)
	ON_BN_CLICKED(IDC_BTN_HOMOGRAPHY, &CMFCApplication1Dlg::OnBnClickedBtnHomography)
END_MESSAGE_MAP()


// CMFCApplication1Dlg 消息处理程序

BOOL CMFCApplication1Dlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码
	namedWindow("view", CV_WINDOW_KEEPRATIO);
	HWND hWnd = (HWND)cvGetWindowHandle("view");
	HWND hParent = ::GetParent(hWnd);
	::SetParent(hWnd, GetDlgItem(IDC_PIC_BOX)->m_hWnd);
	::ShowWindow(hParent, SW_HIDE);
	m_is_from_camera = false;
	m_status = _T("Prepared to work");
	UpdateData(false);
	//CRect rect;
	//GetDlgItem(IDC_PIC_BOX)->GetClientRect(&rect);
	//Rect dst(rect.left, rect.top, rect.right, rect.bottom);


	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CMFCApplication1Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialog::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CMFCApplication1Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialog::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CMFCApplication1Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CMFCApplication1Dlg::OnBnClickedOk()
{
	OpencvHelper opencvHelper;
	// TODO: 在此添加控件通知处理程序代码
	CDialog::OnOK();
}


void CMFCApplication1Dlg::OnBnClickedBtnchoose()
{
	// TODO: 在此添加控件通知处理程序代码
	CString strFile = _T("");

	CFileDialog    dlgFile(TRUE, NULL, NULL, OFN_HIDEREADONLY, _T("JPG Files (*.jpg)|*.jpg|All Files (*.*)|*.*||"), NULL);

	if (dlgFile.DoModal())
	{
		strFile = dlgFile.GetPathName();
		string pic_path = CT2A(strFile.GetBuffer());
		opencv_helper.src_image_ = imread(pic_path);
		Result result = opencv_helper.GetCropRows();
		//AngleResult re;
		/*re.angle_ = result.angle;
		re.offset_ = result.offset;
		CString angle_offset;
		angle_offset.Format(_T("%f;%f \r\n"), re.angle_, re.offset_);
		AddText(angle_offset);*/
		ShowImage(opencv_helper.src_image_);

		m_status = strFile ;
		m_status.Append(_T(" has been choosed"));
	}
	UpdateData(false);
}


void CMFCApplication1Dlg::OnBnClickedIsFromCamera()
{
	UpdateData(true);
	if (m_is_from_camera)
	{
		SetDlgItemText(IDC_SHOWSTATUS, _T("camera is on"));
	}
	else SetDlgItemText(IDC_SHOWSTATUS, _T("camera is off"));
	opencv_helper.homography_ = Homography;
	//opencv_helper.GetImage(m_is_from_camera, paused);
	VideoCapture cap;
	Mat tem_image;
	Rect ROI;
	AngleResult re;
	cap.open(0);//0 for computer while 1 for USB camera.
	//cap.open("E:\\secondyear\\SampleImages\\CH3.avi");
	//cap.open("C:\\Users\\ahaiya\\Desktop\\CH.avi");
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 1024);
	cap.set(CV_CAP_PROP_FPS, 30);
	cap.set(CV_CAP_PROP_BRIGHTNESS, 0);
	cap.set(CV_CAP_PROP_CONTRAST, 0);
	cap.set(CV_CAP_PROP_SATURATION, 64);
	cap.set(CV_CAP_PROP_HUE, 0);

	if (!cap.isOpened())
	{
		cout << "不能初始化摄像头\n";
		waitKey(0);
	}
	//  //方法2  
	while (m_is_from_camera)
	{

		if (!paused)
		{

			cap >> tem_image;
			if (tem_image.empty())
				break;
			else
			{
				double dur;
				clock_t start, end;
				start = clock();

				//ROI = Rect(tem_image.cols / 4, tem_image.rows / 4, tem_image.cols / 2, tem_image.rows / 2);
				ROI = Rect(0, 0,tem_image.cols / 2, tem_image.rows / 2);
				namedWindow("src", CV_WINDOW_KEEPRATIO);
				imshow("src", tem_image);
				//warpPerspective(tem_image, tem_image, Homography, Size(300, 260), CV_INTER_LINEAR);
				//tem_image=tem_image(ROI);
				opencv_helper.SetImage(tem_image);
				Result result = opencv_helper.GetCropRows();
				re.angle_ =int( result.angle);
				re.offset_ = int(result.offset);
				CString angle_offset;
				angle_offset.Format(_T("%d;%d \r\n"), re.angle_, re.offset_);
				AddText(angle_offset);
				UpdateData(TRUE);
				re.UpDate();
				cout << re.can_data_ << endl;
				SendCANMessage(re);
				end = clock();
				dur = (double)(end - start)/ CLOCKS_PER_SEC;
				cout << dur << endl;
				//cv::namedWindow("src", CV_WINDOW_KEEPRATIO);
				cv::imshow("view", opencv_helper.src_image_);    //显示当前帧
			}
			//Result result=opencv_helper.GetCropRows();
			//re.angle_ = result.angle;
			//re.offset_=result.offset;
			//CString angle_offset;
			//angle_offset.Format(_T("%f;%f \r\n"), re.angle_, re.offset_);
			//AddText(angle_offset);
			////UpdateData(TRUE);
			////re.UpDate();
			//cout<<re.can_data_<<endl;
			//SendCANMessage(re);
			//cv::imshow("view", opencv_helper.src_image_);    //显示当前帧
			waitKey(30);  //延时,ms 
		}
		else break;
	}

	//UpdateData(false);
}
void CMFCApplication1Dlg::AddText(CString message)
{
	CEdit* pEdit = (CEdit*)GetDlgItem(IDC_EDIT1);
	int nLength = pEdit->GetWindowTextLength();

	//选定当前文本的末端  
	pEdit->SetSel(nLength, nLength);
	//l追加文本  
	pEdit->ReplaceSel(message);
}

void CMFCApplication1Dlg::OnBnClickedBtnpause()
{
	// TODO: 在此添加控件通知处理程序代码
	if (m_is_from_camera)
	{
		paused = !paused;
		if (paused)
			SetDlgItemText(IDC_SHOWSTATUS, _T("camera is paused"));
		else SetDlgItemText(IDC_SHOWSTATUS, _T("camera is running"));
		opencv_helper.GetImage(m_is_from_camera, paused);		
	}
	else SetDlgItemText(IDC_SHOWSTATUS, _T("Camera is off, turn on the camera first"));
}


void CMFCApplication1Dlg::OnBnClickedProceed()
{
	// TODO: 在此添加控件通知处理程序代码
	VideoCapture captrue(1);
	//视频写入对象  
	VideoWriter writer;
	//写入视频文件名  
	string outFlie = "C:\\Users\\ahaiya\\Desktop\\CH.avi";
	//获得帧的宽高  
	int w = static_cast<int>(captrue.get(CV_CAP_PROP_FRAME_WIDTH));
	int h = static_cast<int>(captrue.get(CV_CAP_PROP_FRAME_HEIGHT));
	Size S(w, h);
	//获得帧率  
	double r = captrue.get(CV_CAP_PROP_FPS);
	//打开视频文件，准备写入  
	writer.open(outFlie, -1, r, S, true);
	//打开失败  
	if (!captrue.isOpened())
	{
		int c= captrue.isOpened();
	}
	bool stop = false;
	Mat frame;
	//循环  
	while (!stop)
	{
		//读取帧  
		if (!captrue.read(frame))
			break;
		imshow("Video", frame);
		//写入文件  
		writer.write(frame);
		//frame.release();
		if (waitKey(10) > 0)
		{
			stop = true;
		}
	}
	//释放对象  
	captrue.release();
	writer.release();
	cvDestroyWindow("Video");
}


void CMFCApplication1Dlg::OnBnClickedBtnsave()
{
	try 
	{
		CString strFile = _T("");
		CFileDialog    dlgFile(TRUE, NULL, NULL, OFN_HIDEREADONLY, _T("JPG Files (*.jpg)|*.jpg|All Files (*.*)|*.*||"), NULL);

		if (dlgFile.DoModal())
		{
			strFile = dlgFile.GetPathName();
			string pic_path = CW2A(strFile.GetBuffer());//CW2A??CT2A.
			
			int save_success = 0;
			if (pic_path.length() != 0)
				save_success = opencv_helper.Save(pic_path);
			if (save_success)
			SetDlgItemText(IDC_SHOWSTATUS, _T("保存成功"));
			else SetDlgItemText(IDC_SHOWSTATUS, _T("保存失败"));

		}
		strFile.ReleaseBuffer();
	}
	// TODO: 在此添加控件通知处理程序代码
	catch (Exception ex)
	{
		MessageBox(_T("迷了"), _T("警告"), MB_OK | MB_ICONQUESTION);
	}

}
void CMFCApplication1Dlg:: ShowImage(Mat image)
{
	Mat imagedst;
	//以下操作获取图形控件尺寸并以此改变图片尺寸  
	CRect rect;
	GetDlgItem(IDC_PIC_BOX)->GetClientRect(&rect);
	Rect dst(rect.left, rect.top, rect.right, rect.bottom);
	resize(image, imagedst, cv::Size(rect.Width(), rect.Height()));//否则会出错
	imshow("view", imagedst);
	//设置显示
	//SetDlgItemText(IDC_SHOWSTATUS, strFile);
}

void CMFCApplication1Dlg::OnBnClickedCancel()
{
	// TODO: 在此添加控件通知处理程序代码
	CDialog::OnCancel();
}


void CMFCApplication1Dlg::OnBnClickedQuit()
{
	 //TODO: 在此添加控件通知处理程序代码
	if (m_is_from_camera)
	{
		m_is_from_camera = false;
		opencv_helper.GetImage(m_is_from_camera, paused);
	}
	CDialog::OnCancel();
	//HomoInitial(1);
}

void CMFCApplication1Dlg::OnBnClickedBtncan()
{
	// TODO: 在此添加控件通知处理程序代码
}

void CMFCApplication1Dlg::OnBnClickedBtnStartcan()
{
	// TODO: 在此添加控件通知处理程序代码
	
}
void CMFCApplication1Dlg::SendCANMessage(AngleResult re)
{
	if (m_connect == 0)
		return;
	VCI_CAN_OBJ frameinfo;
	unsigned char FrameID[4];
	memcpy(&FrameID, re.can_frame_ID, 4);
	BYTE datalen = 8;
	frameinfo.DataLen = datalen;
	memcpy(&frameinfo.Data, re.can_data_, datalen);
	frameinfo.RemoteFlag = 0;
	frameinfo.ExternFlag = 0;
	if (frameinfo.ExternFlag == 1)
	{
		frameinfo.ID = ((DWORD)FrameID[0] << 24) + ((DWORD)FrameID[1] << 16) + ((DWORD)FrameID[2] << 8) +
			((DWORD)FrameID[3]);
	}
	else
	{
		frameinfo.ID = ((DWORD)FrameID[2] << 8) + ((DWORD)FrameID[3]);
	}
	frameinfo.SendType = 0;
	UpdateData(false);
	if (VCI_Transmit(m_devtype, m_devind, m_cannum, &frameinfo, 1) == 1)
	{
		//m_sendcnt += 1;
		ShowInfo("写入成功", 0);
		//SetDlgItemText(IDC_SHOWSTATUS, _T(""));
	}
	else
	{
		ShowInfo("写入失败", 2);
	}
}
void CMFCApplication1Dlg::ShowInfo(const char* str, int code)
{
	CString message;
	message = str;
	//m_ListInfo.InsertString(m_ListInfo.GetCount(), message);
	//m_ListInfo.SetCurSel(m_ListInfo.GetCount() - 1);
}

void CMFCApplication1Dlg::OnBnClickedBtnConectcan()
{
	// TODO: 在此添加控件通知处理程序代码
	if (m_connect == 1)
	{
		m_connect = 0;
		Sleep(500);
		GetDlgItem(IDC_BTN_CONECTCAN)->SetWindowText(_T("连接"));
		VCI_CloseDevice(m_devtype, m_devind);

		KillTimer(ID_TIMER_SHOWCNT);
		return;
	}

	VCI_INIT_CONFIG init_config;
	int index=0, filtertype=0, mode=0, cannum=0;
	m_devtype = 4;//#define VCI_USBCAN1		3    //#define VCI_USBCAN2		4
	CString strcode, strmask, strtiming0, strtiming1, strtmp;
	char szcode[20], szmask[20], sztiming0[20], sztiming1[20];
	unsigned char sztmp[4];
	DWORD code=0, mask=0xffffffff, timing0=0x01, timing1=0x1C;//波特率为250kb时
	init_config.AccCode = code;
	init_config.AccMask = mask;
	init_config.Filter = filtertype;
	init_config.Mode = mode;
	init_config.Timing0 = (UCHAR)timing0;
	init_config.Timing1 = (UCHAR)timing1;

	if (VCI_OpenDevice(m_devtype, index, 0) != STATUS_OK)
	{
		MessageBox(_T("打开设备失败!"), _T("警告"), MB_OK | MB_ICONQUESTION);
		return;
	}
	if (VCI_InitCAN(m_devtype, index, cannum, &init_config) != STATUS_OK)
	{
		MessageBox(_T("初始化CAN失败!"), _T("警告"), MB_OK | MB_ICONQUESTION);
		VCI_CloseDevice(m_devtype, index);
		return;
	}
	m_connect = 1;
	m_devind = index;
	m_cannum = cannum;
	GetDlgItem(IDC_BTN_CONECTCAN)->SetWindowText(_T("断开"));
	//m_recvcount = 0;
	//m_sendcnt = 0;

	SetTimer(ID_TIMER_SHOWCNT, ID_TIMER_SHOWCNT_TM, NULL);
	AfxBeginThread(ReceiveThread, this);
	VCI_StartCAN(m_devtype, m_devind, m_cannum);	
}
int CMFCApplication1Dlg::strtodata(unsigned char *str, unsigned char *data, int len, int flag)
	{
	unsigned char cTmp = 0;
	int i = 0;
	for (int j = 0; j<len; j++)
	{
		if (chartoint(str[i++], &cTmp))
			return 1;
		data[j] = cTmp;
		if (chartoint(str[i++], &cTmp))
			return 1;
		data[j] = (data[j] << 4) + cTmp;
		if (flag == 1)
			i++;
	}
	return 0;
	}
int CMFCApplication1Dlg::chartoint(unsigned char chr, unsigned char *cint)
{
	unsigned char cTmp;
	cTmp = chr - 48;
	if (cTmp >= 0 && cTmp <= 9)
	{
		*cint = cTmp;
		return 0;
	}
	cTmp = chr - 65;
	if (cTmp >= 0 && cTmp <= 5)
	{
		*cint = (cTmp + 10);
		return 0;
	}
	cTmp = chr - 97;
	if (cTmp >= 0 && cTmp <= 5)
	{
		*cint = (cTmp + 10);
		return 0;
	}
	return 1;
}

UINT CMFCApplication1Dlg::ReceiveThread(void *param)
{
	CMFCApplication1Dlg *dlg = (CMFCApplication1Dlg*)param;
	CListBox *box = (CListBox *)dlg->GetDlgItem(IDC_LIST_INFO);
	VCI_CAN_OBJ frameinfo[50];
	VCI_ERR_INFO errinfo;
	int len = 1;
	int i = 0;
	CString str, tmpstr;
	while (1)
	{
		Sleep(1);
		if (dlg->m_connect == 0)
			break;
		len = VCI_Receive(dlg->m_devtype, dlg->m_devind, dlg->m_cannum, frameinfo, 50, 200);
		if (len <= 0)
		{
			//注意：如果没有读到数据则必须调用此函数来读取出当前的错误码，
			//千万不能省略这一步（即使你可能不想知道错误码是什么）
			VCI_ReadErrInfo(dlg->m_devtype, dlg->m_devind, dlg->m_cannum, &errinfo);
		}
		else
		{
#if 1
			//dlg->m_recvcount += len;
			//str.Format("         本次接收 %d 帧  总接收 %d 帧", len, dlg->m_recvcount);
			//box->InsertString(box->GetCount(), str);
			
#endif
#if 0
			//	static UINT s_id=1;
			for (int i = 0; i<len; i++)
			{
				str.Format("s_id=%d  frameinfo[i].ID=%d\n", s_id, frameinfo[i].ID);
				if (frameinfo[i].ID != s_id)
				{
					str += "err err err---------------------------------err err\r\n";

				}
				box->InsertString(box->GetCount(), str);

				TRACE(str);
				s_id++;
				if (s_id>66)
				{
					s_id = 1;
					str.Format("  66  --------------------------------------\r\n");
					box->InsertString(box->GetCount(), str);
				}
			}
#endif

		}
	}
	return 0;
}

void CMFCApplication1Dlg::OnBnClickedBtnSend()
{
	// TODO: 在此添加控件通知处理程序代码
	AngleResult re(1,5);
	SendCANMessage(re);
	re.angle_ = 2;
	re.offset_ = 6;
	re.UpDate();
}


void CMFCApplication1Dlg::OnBnClickedBtnCalibrate()
{
	// TODO: 在此添加控件通知处理程序代码
	string src_path = "E:\\secondyear\\SampleImages\\crops002.jpg";
	Mat src = imread(src_path);
	//namedWindow("calibration", CV_WINDOW_AUTOSIZE);
	if (src.data)
	{
		CameraParams cameraparams(src);
		cameraparams.GetParams();
		SetDlgItemText(IDC_SHOWSTATUS, _T("畸变矫正完毕"));
	}
}


void CMFCApplication1Dlg::OnBnClickedBtnHomography()
{
	// TODO: 在此添加控件通知处理程序代码

	string src_path = "E:\\secondyear\\SampleImages\\crops002.jpg";
	Mat src = imread(src_path);
	//namedWindow("calibration", CV_WINDOW_AUTOSIZE);
	if (src.data)
	{
		CameraParams cameraparams(src);
		Homography=cameraparams.WrapMatrix();
		//HomoInitial(0);
		SetDlgItemText(IDC_SHOWSTATUS, _T("投影矫正完毕"));
	}

}
void HomoInitial(int OperationFlag)
{
	char pFileName[MAX_PATH];
	//LPWSTR pFileName;
	int nPos = GetCurrentDirectory(MAX_PATH, (LPWSTR)pFileName);

	CString csFullPath;
	csFullPath.Format(_T("%s"),pFileName);
	CString sFilePath = csFullPath + _T("\\initial.INI");
	int s = sizeof(Homography);
	if (OperationFlag == 0)

		WritePrivateProfileStruct(_T("VisionNavigation"), _T("Homoraphy"), &Homography, sizeof(Homography), sFilePath);	
	else
	{
		
		GetPrivateProfileStruct(_T("VisionNavigation"), _T("Homoraphy"), &Homography, sizeof(Homography), sFilePath);
	}
	s = sizeof(Homography);
	Homography.at<double>(1, 1);
}
