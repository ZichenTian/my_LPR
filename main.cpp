#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
using namespace cv;
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
  
  Mat element_erode = getStructuringElement(cv::MORPH_ELLIPSE, Size(0.02*width,0.02*height));
  Mat element_dilate = getStructuringElement(cv::MORPH_ELLIPSE, Size(0.02*width,0.02*height));
  
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

Mat getRGBMask4Text(Mat & srcImage)
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
  
  
  Mat element_erode = getStructuringElement(cv::MORPH_ELLIPSE, Size(3,3));
  Mat element_dilate = getStructuringElement(cv::MORPH_ELLIPSE, Size(3,3));
  erode(rgbMask, rgbMask, element_erode);	//腐蚀
  dilate(rgbMask, rgbMask, element_dilate);	//膨胀
  
  
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
  
  Mat element_erode = getStructuringElement(cv::MORPH_ELLIPSE, Size(1,3));		//垂直开操作，消除没去掉的横线
  Mat element_dilate = getStructuringElement(cv::MORPH_ELLIPSE, Size(1,3));
  erode(rgbMask, rgbMask, element_erode);	//腐蚀
  dilate(rgbMask, rgbMask, element_dilate);	//膨胀
  
  element_erode = getStructuringElement(cv::MORPH_ELLIPSE, Size(1,4));		//垂直闭操作，补上被分割的字符
  element_dilate = getStructuringElement(cv::MORPH_ELLIPSE, Size(1,4));
  dilate(rgbMask, rgbMask, element_dilate);	//膨胀
  //erode(rgbMask, rgbMask, element_erode);	//腐蚀
  
  imshow("rgbMask2", rgbMask); waitKey();
  
  //垂直边缘去除
  int thres = 0.02*width;
  for(int j = 0; j < thres; j++)
    for(int i = 0; i < height; i++)
      rgbMask.at<uchar>(i,j) = 0;
   for(int j = width-1; j >= width - thres; j--)
    for(int i = 0; i < height; i++)
      rgbMask.at<uchar>(i,j) = 0;
  
  imshow("rgbMask2", rgbMask); waitKey();
  
  
}

bool cmp(Rect a, Rect b)
{
  return a.area() > b.area();
}

vector<Mat> Split_Text(Mat & rgbMask)
{
  int height = rgbMask.rows;
  int width = rgbMask.cols;
  Mat Mask = rgbMask.clone();
  Mat element_erode = getStructuringElement(cv::MORPH_ELLIPSE, Size(width*0.02,height*0.06));	//膨胀，将断裂的字符连接起来
  Mat element_dilate = getStructuringElement(cv::MORPH_ELLIPSE, Size(width*0.02,height*0.06));
  dilate(Mask, Mask, element_dilate);	//膨胀
  //erode(rgbMask, rgbMask, element_erode);	//腐蚀
  
  imshow("Mask", Mask); waitKey();
  
  
  
  
  vector<vector<Point> > contours;
  findContours(Mask.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
  
  vector<Rect> candidates;
  
  for(int i = 0; i < contours.size(); i++)
  {
    Rect rect = boundingRect(contours[i]);
    double wh_ratio = double(rect.width)/rect.height;
    if(wh_ratio < 1 && wh_ratio > 0.1 && rect.width > 0.1*width && rect.height > 0.2*height)
    {
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
    sort(candidates.begin(), candidates.end(), cmp);
    for(int i = 0; i < 7; i++)
    {
      double wh_ratio = (double)candidates[i].width/candidates[i].height;
      double standard_wh_ratio = 0.5;
      if(wh_ratio < standard_wh_ratio) 	//字符偏瘦，尤其是字符"1"
      {
	candidates[i].x -= candidates[i].width * ((standard_wh_ratio-wh_ratio)/2);
	candidates[i].width += candidates[i].width * (standard_wh_ratio-wh_ratio);
      }
      
      Mat tmp;
      resize(rgbMask(candidates[i]), tmp, Size(13,26), 0, 0, CV_INTER_LINEAR);
      resultText.push_back(tmp);
      imshow("text", resultText[i]); waitKey();
    }
  }
  return resultText;
  
}


int main(void)
{
  Mat srcImage = imread("../detect_picture/011.jpg");
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
  Mat rgbMask4Text = getRGBMask4Text(correctImage);
  Cut_Edge(correctImage, rgbMask4Text); 
  Split_Text(rgbMask4Text);
  
  
  
  
}