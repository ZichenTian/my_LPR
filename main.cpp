#include <opencv2/opencv.hpp>
#include <iostream>
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
  
  Mat element_erode = getStructuringElement(cv::MORPH_ELLIPSE, Size(5,5));
  Mat element_dilate = getStructuringElement(cv::MORPH_ELLIPSE, Size(8,8));
  
  //进行一次开操作消除杂碎点
  erode(hsvMask, hsvMask, element_erode);	//腐蚀
  dilate(hsvMask, hsvMask, element_dilate);	//膨胀
  
  
  imshow("hsvMask", hsvMask); waitKey();
  return hsvMask;
  
  
  
}

Mat getSuspendEdge(Mat& hsvMask, Mat& cannyImage)
{
  int height = hsvMask.rows;
  int width = hsvMask.cols;
  Mat suspendEdge = Mat::zeros(hsvMask.size(), hsvMask.type());
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
  imshow("suspendEdge", suspendEdge);waitKey();
  return suspendEdge;
}



int main(void)
{
  Mat srcImage = imread("/home/tzc/004.jpeg");
  imshow("srcImage", srcImage); waitKey();
  Mat hsvMask = getHSVMask(srcImage);		//筛选符合HSV条件的区域，并进行一次开运算消除杂碎点
  Mat grayImage;
  cvtColor(srcImage, grayImage, CV_RGB2GRAY);
  GaussianBlur(grayImage, grayImage, Size(3,3), 0,0,cv::BORDER_DEFAULT);
  Mat cannyImage;
  int edgeThresh = 50;
  Canny(grayImage, cannyImage, edgeThresh, edgeThresh*3, 3);
  imshow("cannyImage", cannyImage);waitKey();
  Mat suspendEdge = getSuspendEdge(hsvMask, cannyImage);
  
  Mat element_erode = getStructuringElement(cv::MORPH_ELLIPSE, Size(10,10));
  Mat element_dilate = getStructuringElement(cv::MORPH_ELLIPSE, Size(10,10));
  
  dilate(suspendEdge, suspendEdge, element_dilate);	//膨胀
  erode(suspendEdge, suspendEdge, element_erode);	//腐蚀
  
  imshow("suspendEdge", suspendEdge); waitKey();
  
  
  
}


/*
 #include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

void Channel_Split(Mat srcImage, Mat& img_h, Mat& img_s, Mat& img_v)
{
  Mat imghsv;
  vector<Mat> hsv_vec;
  cvtColor(srcImage, imghsv, CV_BGR2HSV);
  imshow("hsv", imghsv);
  waitKey();
  split(imghsv, hsv_vec);
  img_h = hsv_vec[0];
  img_s = hsv_vec[1];
  img_v = hsv_vec[2];
  img_h.convertTo(img_h, CV_32F);
  img_s.convertTo(img_s, CV_32F);
  img_v.convertTo(img_v, CV_32F);
  
  double max_s, max_h, max_v;
  minMaxIdx(img_h, 0, &max_h);
  minMaxIdx(img_s, 0, &max_s);
  minMaxIdx(img_v, 0, &max_v);
  
  img_h /= max_h;
  img_s /= max_s;
  img_v /= max_v;
  
}

Mat Suspend_Point(Mat & img_h, Mat & img_s, Mat & img_v, Mat & cannyImage)
{
  Mat bw_blue = ((img_h > 0.55) & (img_h < 0.625) & (img_s > 0.15) & (img_v > 0.25));
  int height = bw_blue.rows;
  int width = bw_blue.cols;
  imshow("bw_blue", bw_blue);
  Mat bw_blue_edge = Mat::zeros(bw_blue.size(), bw_blue.type());
  for(int k = 1; k != height-2; k++)
  {
    for(int l = 1; l != width-2; l++)
    {
      Rect rct;
      rct.x = l-1;
      rct.y = k-1;
      rct.height = 3;
      rct.width = 3;
      if((cannyImage.at<uchar>(k,l) == 255) && (cv::countNonZero(bw_blue(rct)) >= 1))
	bw_blue_edge.at<uchar>(k,l) = 255;
    }
  }
  imshow("bw_blue_edge", bw_blue_edge);
  return bw_blue_edge;
  
  
}

void Area_Get(Mat & srcImage, Mat & bw_blue_edge)
{
  Mat morph;
  morphologyEx(bw_blue_edge, morph, cv::MORPH_CLOSE, cv::Mat::ones(10,10,CV_8UC1));
  Mat imshow5;
  resize(bw_blue_edge, imshow5, cv::Size(), 1, 1);
  imshow("morphology_bw_blue_edge", imshow5);
  waitKey();
  
  imshow("morph", morph);
  
  vector<vector<Point> > region_contours;
  findContours(morph.clone(), region_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
  
  vector<Rect> candidates;
  vector<Mat> candidates_img;
  Mat result;
  for(size_t n = 0; n != region_contours.size(); ++n)
  {
    Rect rect = boundingRect(region_contours[n]);
    int sub = cv::countNonZero(morph(rect));
    double ratio = double(sub) / rect.area();
    double wh_ratio = double(rect.width) / rect.height;
    if(ratio > 0.3 && wh_ratio > 2 && wh_ratio < 5 && rect.height > 12 && rect.width > 60)
    {
      Mat small = bw_blue_edge(rect);
      result = srcImage(rect);
      imshow("rect", srcImage(rect));
      waitKey();
    }
  }
  
}

void Area_RD_Get(Mat & srcImage, Mat & bw_blue_edge)
{
  Mat morph;
  morphologyEx(bw_blue_edge, morph, cv::MORPH_CLOSE, cv::Mat::ones(2,25,CV_8UC1));
  Mat imshow5;
  resize(bw_blue_edge, imshow5, cv::Size(), 1, 1);
  imshow("morphology_bw_blue_edge", imshow5);
  waitKey();
  
  imshow("morph", morph);
  
  vector<vector<Point> > region_contours;
  findContours(morph.clone(), region_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
  
  vector<Rect> candidates;
  vector<Mat> candidates_img;
  Mat result;
  for(size_t n = 0; n != region_contours.size(); ++n)
  {
    RotatedRect rect = minAreaRect(Mat(region_contours[n]));
    Point2f rect_points[4];
    rect.points(rect_points);
    for(int i = 0; i < 4; i++)
    {
      line(srcImage, rect_points[i], rect_points[(i+1)%4], Scalar(0,0,255), 3, 8);
    }
    imshow("detect", srcImage);
    
    //int sub = cv::countNonZero(morph(rect));
    //double ratio = double(sub) / (rect.size.width * rect.size.height);
    //double wh_ratio = double(rect.size.width) / rect.size.height;
    //if(ratio > 0.5 && wh_ratio > 2 && wh_ratio < 5 && rect.size.width > 12 && rect.size.height > 60)
    //{
      //Mat small = bw_blue_edge(rect);
      //result = srcImage(rect);
      //imshow("rect", srcImage(rect));
      //waitKey();
    //}
  }
}



int main(void)
{
  Mat srcImage = imread("/home/tzc/003.jpeg");
  imshow("srcImage", srcImage);
  Mat grayImage;
  cvtColor(srcImage, grayImage, CV_BGR2GRAY);
  imshow("grayImage", grayImage);
  Mat cannyImage;
  int edgeThresh = 50;
  GaussianBlur(grayImage, grayImage, Size(3,3), 0,0,cv::BORDER_DEFAULT);
  Canny(grayImage, cannyImage, edgeThresh, edgeThresh*3, 3);
  imshow("cannyImage", cannyImage);
  waitKey();
  
  
  
  Mat img_h, img_s, img_v;
  Channel_Split(srcImage, img_h, img_s, img_v);
  Mat bw_blue_edge = Suspend_Point(img_h, img_s, img_v, cannyImage);
  
  Area_Get(srcImage, bw_blue_edge);
  waitKey();
  return 0;
  
  
  return 0;
}
 */

