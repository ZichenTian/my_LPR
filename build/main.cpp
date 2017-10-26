#include <opencv2/opencv.hpp>
#include <iostream>
#include <dirent.h>
#include <algorithm>
#include <string>
using namespace std;
using namespace cv;
using namespace ml;

void calcGradientFeat(const Mat& imgSrc, vector<float>& feat) 
{ 
     float sumMatValue(const Mat& image); // 计算图像中像素灰度值总和 
     
     Mat image; 
     cvtColor(imgSrc,image,CV_BGR2GRAY); 
     resize(image,image,Size(8,16)); 
     Mat tmp = image.clone();
     
     // 计算x方向和y方向上的滤波 
     float mask[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };
 
     Mat y_mask = Mat(3, 3, CV_32F, mask) / 8; 
     Mat x_mask = y_mask.t(); // 转置 
     Mat sobelX, sobelY;
 
     filter2D(image, sobelX, CV_32F, x_mask); 
     filter2D(image, sobelY, CV_32F, y_mask);
 
     sobelX = abs(sobelX); 
     sobelY = abs(sobelY);
 
     float totleValueX = sumMatValue(sobelX); 
     float totleValueY = sumMatValue(sobelY);
 
     // 将图像划分为4*2共8个格子，计算每个格子里灰度值总和的百分比 
     for (int i = 0; i < image.rows; i = i + 4) 
     { 
         for (int j = 0; j < image.cols; j = j + 4) 
         { 
             Mat subImageX = sobelX(Rect(j, i, 4, 4)); 
             feat.push_back(sumMatValue(subImageX) / totleValueX); 
             Mat subImageY= sobelY(Rect(j, i, 4, 4)); 
             feat.push_back(sumMatValue(subImageY) / totleValueY); 
         } 
     }
     
     
     for(int i = 0; i < tmp.rows; i++)
     {
       for(int j = 0; j < tmp.cols; j++)
       {
	 feat.push_back(tmp.at<uchar>(i,j));
       }
     }
 } 
 
 float sumMatValue(const Mat& image) 
 { 
     float sumValue = 0; 
     int r = image.rows; 
     int c = image.cols; 
     if (image.isContinuous()) 
     { 
         c = r*c; 
         r = 1;    
     } 
     for (int i = 0; i < r; i++) 
     { 
         const uchar* linePtr = image.ptr<uchar>(i); 
         for (int j = 0; j < c; j++) 
         { 
             sumValue += linePtr[j]; 
         } 
     } 
     return sumValue; 
 }
 
 vector<float> getOutput(char char_name)
 {
   vector<float> result;
   for(int i = 0; i < 34; i++)
     result.push_back(0);
   if(char_name>='0' && char_name<='9')
     result[char_name-'0'] = 1;
   else if(char_name>='A' && char_name <= 'H')
     result[char_name-'A'+10] = 1;
   else if(char_name>='J' && char_name<='N')
     result[char_name-'J'+10+8] = 1;
   else
     result[char_name-'P'+10+8] = 1;
   
   
   return result;
 }
 
 const int TRAIN_NUM = 50;
 
 float training_data[34*TRAIN_NUM][48];
 float training_result[34*TRAIN_NUM][34];
 

 
int main(void)
{
  Ptr<ANN_MLP> ann = ANN_MLP::create();
  Mat layerSizes = (Mat_<int>(1,3) << 48,48,34);
  ann->setLayerSizes(layerSizes);
  ann->setActivationFunction(ANN_MLP::SIGMOID_SYM);
  ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 50000, 0.001));
  ann->setTrainMethod(ANN_MLP::BACKPROP, 0.1, 0.1);
  
  int count = 0;
  
  
    string root = "../charSamples";
    dirent* ptr;
    DIR* dir = opendir(root.c_str());
    
    vector<string> sub_dir;
    
    while ((ptr=readdir(dir)) != NULL)  
    {  
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir  
                continue;  
        else if(ptr->d_type == 8)    ///file  
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);  
            sub_dir.push_back(ptr->d_name);  
        else if(ptr->d_type == 10)    ///link file  
            //printf("d_name:%s/%s\n",basePath,ptr->d_name);  
            continue;  
        else if(ptr->d_type == 4)    ///dir  
        {  
            sub_dir.push_back(ptr->d_name);  
        }  
    }  
    closedir(dir);
    sort(sub_dir.begin(), sub_dir.end());
    
    for(int i = 0; i < sub_dir.size(); i++)
    {
      int cnt = 0;
      char char_name = (sub_dir[i])[0];
      string pwd = root + "/" + sub_dir[i];
      dir = opendir(pwd.c_str());
      while (cnt < TRAIN_NUM)  
      {  
	ptr=readdir(dir);
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir  
                continue;  
        else if(ptr->d_type == 8)    ///file  
	{
	  string file_name = ptr->d_name;
	  file_name = pwd + "/" + file_name;
	  Mat srcImage = imread(file_name);
	  //imshow("srcImage", srcImage); waitKey();
	  vector<float> feat;
	  calcGradientFeat(srcImage, feat);
	  
	  for(int i = 0; i < 48; i++)
	  {
	    training_data[count][i] = feat[i];
	  }
	  vector<float> outPut = getOutput(char_name);
	  for(int i = 0; i < 34; i++)
	  {
	    training_result[count][i] = outPut[i];
	  }
	  count++;
	  cnt++;
	  
	}
      }
      closedir(dir);
      
    }
    
    Mat training_data_mat(34*TRAIN_NUM,48, CV_32FC1, training_data);
    Mat training_result_mat(34*TRAIN_NUM,34, CV_32FC1, training_result);
    
    Ptr<TrainData> tData = TrainData::create(training_data_mat, ROW_SAMPLE, training_result_mat);
    ann->train(tData);
    
    Mat testMat = imread("../10.png");
    imshow("test", testMat); waitKey();
    vector<float> feat;
    calcGradientFeat(testMat, feat);
    
    float test_input[1][48];
    for(int i = 0; i < 48; i++)
    {
      test_input[0][i] = feat[i];
    }
    Mat test_input_mat(1,48, CV_32FC1, test_input);
    //cout << test_input_mat << endl;
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
  
  
  
}


 
 /*

int main(void)
{
  Ptr<ANN_MLP> ann = ANN_MLP::create();
  Mat layerSizes = (Mat_<int>(1,3) << 2,3,1);
  ann->setLayerSizes(layerSizes);
  ann->setActivationFunction(ANN_MLP::SIGMOID_SYM);
  ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01));
  ann->setTrainMethod(ANN_MLP::BACKPROP, 0.1, 0.1);
  
  const int max = 200;
  
  float training_data[max][2];
  float training_result[max];
  for(int i = 0; i < max; i++)
  {
    long long a, b;
    a = rand();
    b = rand();
    double a1, b1;
    a1 = (double)a/RAND_MAX;
    b1 = (double)b/RAND_MAX;
    training_data[i][0] = a1;
    training_data[i][1] = b1;
    training_result[i] = a1>b1?1:0;
  }
  
  Mat training_data_mat(max,2, CV_32FC1, training_data);
  Mat training_result_mat(max,1, CV_32FC1, training_result);
  
  Ptr<TrainData> tData = TrainData::create(training_data_mat, ROW_SAMPLE, training_result_mat);
  ann->train(tData);
  
  Mat test_input = (Mat_<float>(1,2) << 0.9,0.6);
  Mat test_output;
  ann->predict(test_input, test_output);
  std::cout << test_output << std::endl;
  
}
*/
