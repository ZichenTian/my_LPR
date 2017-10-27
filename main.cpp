#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
using namespace cv;
using namespace ml;
using namespace std;

Mat getHSVMask(Mat& srcImage)
{
  
 
  int height = srcImage.rows;
  int width = srcImage.cols;
  double h,s,v;
  uchar r,g,b;
  uchar max, min, delta;
  Mat hsvMask = Mat::zeros(srcImage.size(), CV_8UC1);
  
  
  for(int i = 0; i < height; i++)
  {
    for(int j = 0; j < width; j++)
    {
      b = srcImage.at<Vec3b>(i,j)[0];
      g = srcImage.at<Vec3b>(i,j)[1];
      r = srcImage.at<Vec3b>(i,j)[2];
      
      max = (r>g)?r:g;
      max = (max>b)?max:b;
      
      min=(r<g)?r:g;
      min=(min<b)?min:b;
      
      delta = max-min;
      if(delta == 0)
	h = 0;
      else
      {
	if(max==r && g>=b)
	  h = double(60*(g-b))/delta;
	if(max==r && g<b)
	  h = double(60*(g-b))/delta+360;
	if(max==g)
	  h = double(60*(b-r))/delta+120;
	if(max==b)
	  h = double(60*(r-g))/delta+240;
      }
      
      v = max;
      if(v==0)
	s = 0;
      else
	s = double(delta)/max;
      if(h<0)
	h+=360;
      if(h<=270&&h>=208&&s>0.52)
      {
	hsvMask.at<uchar>(i,j) = 255;
      }
      if(h<=270&&h>=205&&s>0.48)
      {
	hsvMask.at<uchar>(i,j) = 255;
      }
      
      
    }
  }
  
  
  
  
  //imshow("hsvMask", hsvMask); waitKey();
  return hsvMask;
  
  
  
}

Mat getSuspendEdge(Mat& hsvMask, Mat& cannyImage)
{
  int height = hsvMask.rows;
  int width = hsvMask.cols;
  Mat suspendEdge = Mat::zeros(hsvMask.size(), hsvMask.type());
  
  Mat element_erode = getStructuringElement(cv::MORPH_ELLIPSE, Size(5,5));
  Mat element_dilate = getStructuringElement(cv::MORPH_ELLIPSE, Size(5,5));
  
  //进行一次开操作消除杂碎点
  erode(hsvMask, hsvMask, element_erode);	//腐蚀
  dilate(hsvMask, hsvMask, element_dilate);	//膨胀
  
  for(int i = 1; i != height-2; i++)
  {
    for(int j = 1; j != width-2; j++)
    {
      Rect rct;
      rct.x = j-1;
      rct.y = i-1;
      rct.height = 3;
      rct.width = 3;
      if((cannyImage.at<uchar>(i,j) == 255) && (cv::countNonZero(hsvMask(rct)) >= 1))
	suspendEdge.at<uchar>(i,j) = 255;
    }
  }
  //imshow("suspendEdge", suspendEdge);waitKey();
  return suspendEdge;
}

Mat getFinalMask(Mat& suspendEdge)
{
  Mat finalMask = suspendEdge.clone(); 
  int height = finalMask.rows;
  int width = finalMask.cols;
  
  Mat element_erode = getStructuringElement(cv::MORPH_ELLIPSE, Size(0.025*width,0.025*height));
  Mat element_dilate = getStructuringElement(cv::MORPH_ELLIPSE, Size(0.025*width,0.025*height));
  
  dilate(finalMask, finalMask, element_dilate);	//膨胀
  erode(finalMask, finalMask, element_erode);	//腐蚀
  
  //element_erode = getStructuringElement(cv::MORPH_ELLIPSE, Size(20,20));
  //element_dilate = getStructuringElement(cv::MORPH_ELLIPSE, Size(20,20));
  
  //erode(finalMask, finalMask, element_erode);	//腐蚀
  //dilate(finalMask, finalMask, element_dilate);	//膨胀
  
  
  imshow("finalMask", finalMask); waitKey();
  return finalMask;
}

Mat getArea(Mat& srcImage, Mat& finalMask)
{
  int height = finalMask.rows;
  int width = finalMask.cols;
  
  vector<vector<Point> > contours;
  findContours(finalMask.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
  
  vector<Rect> candidates;
  int best_i = -1;
  int best_area = 0;
  
  double ratio_high = 0.3;
  double ratio_low = 0.1;
  
  for(int i = 0; i != contours.size(); i++)
  {
    Rect rect = boundingRect(contours[i]);
    int area = countNonZero(finalMask(rect));
    double ratio = double(area)/rect.area();
    double wh_ratio = double(rect.width)/ rect.height;
    if(ratio > ratio_high && wh_ratio > 1.5 && wh_ratio < 5 && rect.height > 12 && rect.width > 60)
    {
      if(area > best_area)
      {
	best_area = area;
	best_i = i;
      }
    }
    else if(ratio > ratio_low && wh_ratio > 1.5 && wh_ratio < 5 && rect.height > 12 && rect.width > 60)
    {
      Mat element_dilate = getStructuringElement(cv::MORPH_ELLIPSE, Size(0.02*width,0.02*height));
      Mat tmp = finalMask.clone();
      dilate(tmp, tmp, element_dilate);	//膨胀
      area = countNonZero(tmp(rect));
      ratio = double(area)/rect.area();
      if(ratio > ratio_high && wh_ratio > 1.5 && wh_ratio < 5 && rect.height > 12 && rect.width > 60)
      {
	if(area > best_area)
	{
	  best_area = area;
	  best_i = i;
	}
      }
    }
    
  }
  if(best_i < 0)
    cout << "no area found!" << endl;
  
  Rect rect = boundingRect(contours[best_i]);
  
  //cout << rect.width << " " << rect.height << endl;
  
  
  //rect.x -= rect.width * 0.05;
  rect.y -= rect.height * 0.05;
  //rect.width += rect.width * 0.1;
  rect.height += rect.height * 0.1;
  
  Mat resultArea = srcImage(rect);
  //resize(resultArea, resultArea, Size(150,50), 0, 0, CV_INTER_LINEAR);
  imshow("resultArea", resultArea); waitKey();
  return resultArea;
      
}




Mat getCorrect(Mat& resultArea)
{
  Mat correctImage = resultArea.clone();
   cvtColor(correctImage, correctImage, CV_BGR2GRAY);
  correctImage.convertTo(correctImage, CV_8UC1);
  Canny(correctImage, correctImage, 50, 200, 3);
  
  vector<Vec4i> lines;
  HoughLinesP(correctImage, lines, 1, CV_PI/180, 50, 50, 10);
  
  
  double max_length = 0;
  int max_i = -1;
  for(int i = 0; i < lines.size(); i++)
  {
    Vec4i  l = lines[i];
    double length = sqrt((l[2]-l[0])*(l[2]-l[0]) + (l[3]-l[1])*(l[3]-l[1]));
    if(length > max_length)
    {
      max_length = length;
      max_i = i;
    }
  }
  Vec4i l = lines[max_i];
  //cout << l[0] <<" " <<  l[1] << " "  << l[2] << " "<< l[3] << endl;
  
  
  //double angle = atan((double)(l[3]-l[1])/(l[2]-l[0]));
  double angle = cvFastArctan(l[3]-l[1], l[2]-l[0]);
  Point2f center = Point2f(l[0], l[1]);
  angle *= CV_PI/180;
  double scale = 1;
  double alpha = cos(angle)*scale;
  double beta = sin(angle)*scale;
  Mat M(2,3,CV_64F);
  double* m = (double*)M.data;
  m[0] = alpha;
  m[1] = beta;
  m[2] = (1-alpha)*center.x - beta*center.y;
  m[3] = -beta;
  m[4] = alpha;
  m[5] = beta*center.x + (1-alpha)*center.y;
  warpAffine(resultArea, resultArea, M, resultArea.size());
  imshow("correctImage", resultArea); waitKey();
  
  return resultArea;
  
  
}

Mat getRGBMask4Text(Mat & srcImage, Mat & rgbMask4Final)
{
  
  Mat rgbMask = getHSVMask(srcImage);
  //imshow("rgbMask", rgbMask);waitKey();
  
   int height = srcImage.rows;
  int width = srcImage.cols;
  uchar r,g,b;
  //Mat rgbMask = Mat::zeros(srcImage.size(), CV_8UC1);
  
  
  for(int i = 0; i < height; i++)
  {
    for(int j = 0; j < width; j++)
    {
      b = srcImage.at<Vec3b>(i,j)[0];
      g = srcImage.at<Vec3b>(i,j)[1];
      r = srcImage.at<Vec3b>(i,j)[2];
      
      int thresh = 40;
      if(abs(r-g) < thresh && abs(r-b) < thresh && abs(b-g) < thresh)
      {
	int thresh2 = 80;
	if(r < thresh2 && g < thresh2 && b < thresh2)
	{
	  rgbMask.at<uchar>(i,j) = 255;
	}
      }
    }
  }
  
  /*
  Mat element_erode = getStructuringElement(cv::MORPH_ELLIPSE, Size(3,3));
  Mat element_dilate = getStructuringElement(cv::MORPH_ELLIPSE, Size(3,3));
  erode(rgbMask, rgbMask, element_erode);	//腐蚀
  dilate(rgbMask, rgbMask, element_dilate);	//膨胀
  */
  
  rgbMask4Final = rgbMask.clone();
  rgbMask4Final = 255 - rgbMask4Final;
  
  GaussianBlur(rgbMask, rgbMask, Size(5,5), 0,0,cv::BORDER_DEFAULT);
  for(int i = 0; i < height; i++)
    for(int j = 0; j < width; j++)
    {
      if(rgbMask.at<uchar>(i,j) > 180)
	rgbMask.at<uchar>(i,j) = 255;
      else
	rgbMask.at<uchar>(i,j) = 0;
    }
	
  
  rgbMask = 255 - rgbMask;
  imshow("rgbMask", rgbMask);waitKey();
  return rgbMask;
}

void Cut_Edge(Mat& srcImage, Mat& rgbMask)
{    
  int height = srcImage.rows;
  int width = srcImage.cols;
  
  //水平边缘去除
  
  
  
  for(int i = 0; i < height; i++)
  {
    int thresh = 15;
    int cnt = 0;
    for(int j = 1; j < width; j++)
    {
      if((rgbMask.at<uchar>(i,j) == 255 && rgbMask.at<uchar>(i,j-1) == 0) || (rgbMask.at<uchar>(i,j) == 0 && rgbMask.at<uchar>(i,j-1) == 255))
	cnt++;
    }
    if(cnt < thresh)
    {
      for(int j = 0; j < width; j++)
	rgbMask.at<uchar>(i,j) = 0;
    
    }
  }
  
  /*
  Mat element_erode = getStructuringElement(cv::MORPH_ELLIPSE, Size(2,5));		//垂直开操作，消除没去掉的横线
  Mat element_dilate = getStructuringElement(cv::MORPH_ELLIPSE, Size(2,5));
  erode(rgbMask, rgbMask, element_erode);	//腐蚀
  dilate(rgbMask, rgbMask, element_dilate);	//膨胀
  */
  Mat element_erode = getStructuringElement(cv::MORPH_ELLIPSE, Size(2,5));		//垂直闭操作，补上被分割的字符
  Mat element_dilate = getStructuringElement(cv::MORPH_ELLIPSE, Size(2,5));
  dilate(rgbMask, rgbMask, element_dilate);	//膨胀
  erode(rgbMask, rgbMask, element_erode);	//腐蚀
  
  imshow("rgbMask2", rgbMask); waitKey();
  
  //垂直边缘去除
  /*
  int thres = 0.02*width;
  for(int j = 0; j < thres; j++)
    for(int i = 0; i < height; i++)
      rgbMask.at<uchar>(i,j) = 0;
   for(int j = width-1; j >= width - thres; j--)
    for(int i = 0; i < height; i++)
      rgbMask.at<uchar>(i,j) = 0;
  
  imshow("rgbMask2", rgbMask); waitKey();
  */
  
}

bool cmp1(Rect a, Rect b)
{
  return a.area() > b.area();
}

bool cmp2(Rect a, Rect b)
{
  return a.x < b.x;
}

vector<Mat> Split_Text(Mat & rgbMask, Mat& rgbMask4Final)
{
  int height = rgbMask.rows;
  int width = rgbMask.cols;
  Mat Mask = rgbMask.clone();
  Mat element_erode = getStructuringElement(cv::MORPH_ELLIPSE, Size(width*0.02,(height*0.001)>3?height:3));	
  Mat element_dilate = getStructuringElement(cv::MORPH_ELLIPSE, Size(width*0.025,(height*0.001)>3?height:3));
  imshow("Mask", Mask); waitKey();
  erode(Mask, Mask, element_erode);		//先水平方向腐蚀，去掉垂直边缘
  imshow("Mask", Mask); waitKey();
  dilate(Mask, Mask, element_dilate);			//水平方向膨胀，并补上断裂字符
  imshow("Mask", Mask); waitKey();
  
  element_erode = getStructuringElement(cv::MORPH_ELLIPSE, Size(((width*0.001)>3?width:3),height*0.03));	
  element_dilate = getStructuringElement(cv::MORPH_ELLIPSE, Size(((width*0.001)>3?width:3),height*0.07));
  
  erode(Mask, Mask, element_erode);	//垂直方向腐蚀，去掉残余的水平边缘
  imshow("Mask", Mask); waitKey();
  dilate(Mask, Mask, element_dilate);		//垂直方向膨胀，补上断裂字符
  
  imshow("Mask", Mask); waitKey();
  
  
  
  
  vector<vector<Point> > contours;
  findContours(Mask.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
  
  vector<Rect> candidates;
  
  for(int i = 0; i < contours.size(); i++)
  {
    Rect rect = boundingRect(contours[i]);
    double wh_ratio = double(rect.width)/rect.height;
    //imshow("test", Mask(rect)); waitKey();
    if(wh_ratio < 1 && wh_ratio > 0.1 && rect.width > 0.03*width && rect.height > 0.1*height)
    {
      rect.x -= rect.width * 0.05;
  rect.y -= rect.height * 0.05;
  rect.width += rect.width * 0.1;
  rect.height += rect.height * 0.1;
  
  if(rect.x < 0)		//防止超出
    rect.x = 0;
  if(rect.y < 0)
    rect.y = 0;
  
  if(rect.width+rect.x >= width)	
    rect.width = width - rect.x-1;
  if(rect.height+rect.y >= height)
    rect.height = height - rect.y-1;
  
      candidates.push_back(rect);
    }
  }
  
  vector<Mat> resultText;
  
  if(candidates.size() < 7)
  {
    cout << "cannot get enough text!" << endl;
    cout << candidates.size() << endl;
  }
  else
  {
    sort(candidates.begin(), candidates.end(), cmp1);		//按大小排序，选出区域大小最大的7块字符
    sort(candidates.begin(), candidates.begin()+7, cmp2);	//7块字符按位置从左到右排序
    for(int i = 0; i < 7; i++)
    {
      double wh_ratio = (double)candidates[i].width/candidates[i].height;
      double standard_wh_ratio = 0.5;      
      if(wh_ratio < standard_wh_ratio) 	//字符偏瘦，尤其是字符"1"
      {
	int wide = (standard_wh_ratio*candidates[i].height - candidates[i].width)/2;
	candidates[i].x -= wide;
	candidates[i].width += 2*wide;
	
	if(candidates[i].x < 0)		//防止超出
	  candidates[i].x = 0;
	if(candidates[i].y < 0)
	  candidates[i].y = 0;
  
	if(candidates[i].width+candidates[i].x >= width)	
	  candidates[i].width = width - candidates[i].x-1;
	if(candidates[i].height+candidates[i].y >= height)
	  candidates[i].height = height - candidates[i].y-1;	
      }
      
      
      
      Mat tmp;
      blur(tmp, tmp, Size(5,5));
      resize(rgbMask4Final(candidates[i]), tmp, Size(16,32), 0, 0, CV_INTER_AREA);
      equalizeHist(tmp, tmp);
      resultText.push_back(tmp);
      //imshow("text", resultText[i]); waitKey();
    }
  }
  return resultText;
  
}

void calcGradientFeat(const Mat& imgSrc, vector<float>& feat) 
{ 
     float sumMatValue(const Mat& image); // 计算图像中像素灰度值总和 
     
     Mat image; 
     if(imgSrc.type() != CV_8UC1)
      cvtColor(imgSrc,image,CV_BGR2GRAY); 
     else
      image = imgSrc.clone();
     resize(image,image,Size(8,16)); 
     Mat tmp = image.clone();
     
     for(int i = 0; i < tmp.rows; i++)
     {
       for(int j = 0; j < tmp.cols; j++)
       {
	 feat.push_back((float)tmp.at<uchar>(i,j));
       }
     }
 }
 
 char translateNum(int num)
 {
   char result;
   if(num<=9)
     result = num+'0';
   else if(num <= 17)
     result = num+'A'-10;
   else if(num <= 22)
     result = num+'A'+1-10;
   else
     result = num+'A'+2-10;
   return result;
 }


int main(void)
{
  
  
  Mat srcImage = imread("../detect_picture/030.jpg");
  //resize(srcImage, srcImage, Size(480, 640), 0, 0, CV_INTER_LINEAR);
  imshow("srcImage", srcImage); waitKey();
  Mat hsvMask = getHSVMask(srcImage);		//筛选符合HSV条件的区域，并进行一次开运算消除杂碎点
  Mat grayImage; 
  cvtColor(srcImage, grayImage, CV_RGB2GRAY);
  GaussianBlur(grayImage, grayImage, Size(3,3), 0,0,cv::BORDER_DEFAULT);
  Mat cannyImage;
  int edgeThresh = 50;
  Canny(grayImage, cannyImage, edgeThresh, edgeThresh*3, 3);
  //imshow("cannyImage", cannyImage);waitKey();
  Mat suspendEdge = getSuspendEdge(hsvMask, cannyImage);
  Mat finalMask = getFinalMask(suspendEdge);
  Mat resultArea = getArea(srcImage, finalMask);
  Mat correctImage = getCorrect(resultArea);
  Mat rgbMask4Final;
  Mat rgbMask4Text = getRGBMask4Text(correctImage, rgbMask4Final);
  Cut_Edge(correctImage, rgbMask4Text); 
  vector<Mat> splitText =  Split_Text(rgbMask4Text, rgbMask4Final);
  
  Ptr<ANN_MLP> ann = ANN_MLP::load<ANN_MLP>("mlp.xml");
  
  for(int i = 1; i < 7; i++)		//汉字不识别
  {
    vector<float> feat;
    imshow("origin", splitText[i]);waitKey();
    calcGradientFeat(splitText[i], feat);
    float test_input[1][128];
    for(int j = 0; j < 128; j++)
      test_input[0][j] = feat[j];
    Mat test_input_mat(1,128,CV_32FC1,test_input);
    Mat test_output_mat;
    ann->predict(test_input_mat, test_output_mat);
    cout << test_output_mat << endl;
    float max = -10;
    int max_x = 0;
    float* p1tr = (float*)test_output_mat.data;
    for(int i = 0; i < 34; i++)
    {
      if(*(p1tr+i) > max)
      {
	max = *(p1tr+i);
	max_x = i;
      }
    }
    cout << max_x << endl;
    cout << translateNum(max_x) << endl;
    
  }
  
  
  
  
}