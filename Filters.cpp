#include "Filters.h"

void WhiteNoise(const cv::Mat& input, cv::Mat& output) {
  std::mt19937 gen(time(0));
  for (int i = 0; i < input.rows; ++i)
    for (int j = 0; j < input.cols; ++j) {
      if (gen() % __CHANCE_ == 0) {
        output.at<cv::Vec3b>(i, j)[0] = 255;
        output.at<cv::Vec3b>(i, j)[1] = 255;
        output.at<cv::Vec3b>(i, j)[2] = 255;
      } else {
        output.at<cv::Vec3b>(i, j)[0] = input.at<cv::Vec3b>(i, j)[0];
        output.at<cv::Vec3b>(i, j)[1] = input.at<cv::Vec3b>(i, j)[1];
        output.at<cv::Vec3b>(i, j)[2] = input.at<cv::Vec3b>(i, j)[2];
      }
    }
}

void MedianFilter(const cv::Mat& input, cv::Mat& output, int radius) { return; }

void ATrimFilter(const cv::Mat& input, cv::Mat& output, int radius, int alpha) {
  {
    int vec_size = pow(2 * radius + 1, 2);
    int vec_resize = vec_size - (alpha /2) * 2;
    for (int i = radius; i < input.rows - radius; ++i) {
      for (int j = radius; j < input.cols - radius; ++j) {
        std::vector<int> arrB(vec_size);
        std::vector<int> arrG(vec_size);
        std::vector<int> arrR(vec_size);
        cv::Vec3b color;
        int count = 0;
        for (int a = -radius; a <= radius; ++a)
          for (int b = -radius; b <= radius; ++b) {
            cv::Vec3b iter_color = input.at<cv::Vec3b>(i + a, j + b);
            arrB[count] = iter_color[0];
            arrG[count] = (iter_color[1]);
            arrR[count] = (iter_color[2]);
            count++;
          }

        std::sort(arrR.begin(), arrR.end());
        std::sort(arrG.begin(), arrG.end());
        std::sort(arrB.begin(), arrB.end());

        int resB = 0, resG = 0, resR = 0;
        for (int i = alpha / 2; i < vec_size - alpha / 2; i++) {
          resB += arrB[i];
          resG += arrG[i];
          resR += arrR[i];
        }
        color[0] = resB / vec_resize;
        color[1] = resG / vec_resize;
        color[2] = resR / vec_resize;

        output.at<cv::Vec3b>(i, j) = color;
      }
    }
  }
}

double ConditionalExp(const cv::Mat& input) {
  double sum = 0;
  int square = input.rows * input.cols;

  for (int i = 0; i < input.rows; ++i)
    for (int j = 0; j < input.cols; ++j) {
      cv::Vec3b color = input.at<cv::Vec3b>(i, j);
      sum += (color[0] + color[1] + color[2]) / 3; 
    }
  return sum / square;
}

double Dispersion(const cv::Mat& input,const double conditional_expectation) {
  double sum = 0;
  int square = input.rows * input.cols;
  for (int i = 0; i < input.rows; ++i)
    for (int j = 0; j < input.cols; ++j) {
      cv::Vec3b color = input.at<cv::Vec3b>(i, j);
      sum += pow((color[0] + color[1] + color[2]) / 3 - conditional_expectation, 2) ;
    }
  return sum / square;
}
