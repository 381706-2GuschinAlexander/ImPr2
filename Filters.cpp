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
    int vec_resize = vec_size - (alpha / 2) * 2;
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

void Monochrome(const cv::Mat& input, cv::Mat& output) {
  for (int i = 0; i < input.rows; ++i)
    for (int j = 0; j < input.cols; ++j) {
      cv::Vec3b colorIn = input.at<cv::Vec3b>(i, j);
      unsigned short colorOut =
          colorIn[0] * 0.0721 + colorIn[1] * 0.7154 + colorIn[2] * 0.2125;
      output.at<cv::Vec3b>(i, j) = cv::Vec3b(colorOut, colorOut, colorOut);
    }
}

unsigned char OtsuThreshold(const cv::Mat& input) {
  int hist[256];
  int sumOfIntensity = 0;
  for (int i = 0; i < 256; ++i) hist[i] = 0;
  for (int i = 0; i < input.rows; ++i)
    for (int j = 0; j < input.cols; ++j) {
      sumOfIntensity += input.at<cv::Vec3b>(i, j)[0];
      hist[input.at<cv::Vec3b>(i, j)[0]]++;
    }

  int pixelCount = input.rows * input.cols;
  int bestThresh = 0;
  double bestSigma = 0.0;
  int firstClassPixelCount = 0;
  int64_t firstClassIntensitySum = 0;
  for (int i = 0; i < 255; ++i) {
    firstClassPixelCount += hist[i];
    firstClassIntensitySum += i * hist[i];
    double firstClassProb =
        firstClassPixelCount / static_cast<double>(pixelCount);
    double secondClassProb = 1.0 - firstClassProb;
    double firstClassMean =
        firstClassIntensitySum / static_cast<double>(firstClassPixelCount);
    double secondClassMean =
        (sumOfIntensity - firstClassIntensitySum) /
        static_cast<double>(pixelCount - firstClassPixelCount);
    double meanDelta = firstClassMean - secondClassMean;
    double sigma = firstClassProb * secondClassProb * pow(meanDelta, 2);
    if (sigma > bestSigma) {
      bestSigma = sigma;
      bestThresh = i;
    }
  }
  return bestThresh;
}

void Binarization(const cv::Mat& input, cv::Mat& output,
                  unsigned char threshold) {
  for (int i = 0; i < input.rows; ++i)
    for (int j = 0; j < input.cols; ++j) {
      unsigned char currColor = input.at<cv::Vec3b>(i, j)[0];
      unsigned char newColor;
      newColor = currColor < threshold ? 0 : 255;
      output.at<cv::Vec3b>(i, j) = cv::Vec3b(newColor, newColor, newColor);
    }
}

void OtsuFilter(const cv::Mat& input, cv::Mat& output) {
  Monochrome(input, output);
  unsigned char threshold = OtsuThreshold(output);
  cv::Mat newInput;
  output.copyTo(newInput);
  Binarization(newInput, output, threshold);
}

void GrowFilter(cv::Mat& image, int thr) {
  Matrix A(image.rows, std::vector<int>(image.cols));
  int count = 0;
  for (int i = 0; i < image.rows; ++i)
    for (int j = 0; j < image.cols; ++j) {
      cv::Vec3b color = image.at<cv::Vec3b>(i, j);
      int I = ToGray(color);

      int I_left = 1000;
      int I_up = 1000;

      if (i > 0) {
        cv::Vec3b tmp_color = image.at<cv::Vec3b>(i - 1, j);
        I_left = ToGray(tmp_color);
      }
      if (j > 0) {
        cv::Vec3b tmp_color = image.at<cv::Vec3b>(i, j - 1);
        I_up = ToGray(tmp_color);
      }

      if (abs(I - I_left) > thr && abs(I - I_up) > thr) {
        count++;
        image.at<cv::Vec3b>(i, j)[0] = I;
        image.at<cv::Vec3b>(i, j)[1] = I;
        image.at<cv::Vec3b>(i, j)[2] = I;
        A[i][j] = count;
      } else if (abs(I - I_left) < thr && abs(I - I_up) < thr && I_left != I_up) {
        if (abs(I_up - I_left) < thr) {
          count++;
          int min = 0;
          Merge(image, A[i - 1][j], A[i][j - 1], I, i, j, A, count);
          image.at<cv::Vec3b>(i, j)[0] = I;
          image.at<cv::Vec3b>(i, j)[1] = I;
          image.at<cv::Vec3b>(i, j)[2] = I;
          A[i][j] = count;
        } else {
          if (abs(I - I_left) < abs(I - I_up)) {
            image.at<cv::Vec3b>(i, j)[0] = I_left;
            image.at<cv::Vec3b>(i, j)[1] = I_left;
            image.at<cv::Vec3b>(i, j)[2] = I_left;
            A[i][j] = A[i - 1][j];
          } else {
            image.at<cv::Vec3b>(i, j)[0] = I_up;
            image.at<cv::Vec3b>(i, j)[1] = I_up;
            image.at<cv::Vec3b>(i, j)[2] = I_up;
            A[i][j] = A[i][j - 1];
          }
        }
      } else if (abs(I - I_left) < thr) {
        image.at<cv::Vec3b>(i, j)[0] = I_left;
        image.at<cv::Vec3b>(i, j)[1] = I_left;
        image.at<cv::Vec3b>(i, j)[2] = I_left;
        A[i][j] = A[i - 1][j];
      } else if (abs(I - I_up) < thr) {
        image.at<cv::Vec3b>(i, j)[0] = I_up;
        image.at<cv::Vec3b>(i, j)[1] = I_up;
        image.at<cv::Vec3b>(i, j)[2] = I_up;
        A[i][j] = A[i][j - 1];
      }

    }
}

void Merge(cv::Mat& image, int reg1, int reg2, int nI, int end_x, int end_y, Matrix& A, int count) { 
  for (int i = 0; i < image.rows; ++i)
    for (int j = 0; j < image.cols; ++j) {
      if (A[i][j] == reg1 || A[i][j] == reg2) {
        image.at<cv::Vec3b>(i, j)[0] = nI;
        image.at<cv::Vec3b>(i, j)[1] = nI;
        image.at<cv::Vec3b>(i, j)[2] = nI;
        A[i][j] = count;
    }
    if (i == end_x && j == end_y) return;
  }
}

//double Distance(const cv::Vec3b&  f_c, const cv::Vec3b& s_c) { 
//  return sqrt(pow(f_c[0] - s_c[0], 2) + pow(f_c[1] - s_c[1], 2) +
//              pow(f_c[2] - s_c[2], 2));
//}

int ToGray(const cv::Vec3b& color) {
  return (color[0] + color[1] + color[2])/3;
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

double Dispersion(const cv::Mat& input, const double conditional_expectation) {
  double sum = 0;
  int square = input.rows * input.cols;
  for (int i = 0; i < input.rows; ++i)
    for (int j = 0; j < input.cols; ++j) {
      cv::Vec3b color = input.at<cv::Vec3b>(i, j);
      sum += pow((color[0] + color[1] + color[2]) / 3 - conditional_expectation,
                 2);
    }
  return sum / square;
}

double covFuncion(double& mW1, double& mW2, cv::Mat& a, cv::Mat& b) {
  double cov = 0;
  for (int i = 0; i < a.rows; i++)
    for (int j = 0; j < a.cols; j++)
      cov = ((a.at<cv::Vec3b>(i, j)[0] + a.at<cv::Vec3b>(i, j)[1] +
              a.at<cv::Vec3b>(i, j)[2]) /
                 3 -
             mW1) *
            ((b.at<cv::Vec3b>(i, j)[0] + b.at<cv::Vec3b>(i, j)[1] +
              b.at<cv::Vec3b>(i, j)[2]) /
                 3 -
             mW2);
  cov = sqrt(cov);
  return cov;
}

double ssim(double& cE1, double& cE2, double& dis1, double& dis2, double cov) {
  return (2 * cE1 * cE2 + 0.0001) * (2 * cov + 0.0001) /
         ((pow(cE1, 2) + pow(cE2, 2) + 0.0001) * (dis1 + dis2 + 0.0001));
}
