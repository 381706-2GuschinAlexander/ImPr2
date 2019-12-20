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

void Monochrome(const cv::Mat& input, cv::Mat& output) {
  for (int i = 0; i < input.rows; ++i)
    for (int j = 0; j < input.cols; ++j) {
      cv::Vec3b colorIn = input.at<cv::Vec3b>(i, j);
      unsigned short colorOut = colorIn[0] * 0.0721 + colorIn[1] * 0.7154 + colorIn[2] * 0.2125;
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
    double firstClassProb = firstClassPixelCount / static_cast<double>(pixelCount);
    double secondClassProb = 1.0 - firstClassProb;
    double firstClassMean = firstClassIntensitySum / static_cast<double>(firstClassPixelCount);
    double secondClassMean = (sumOfIntensity - firstClassIntensitySum) /
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

void Binarization(const cv::Mat& input, cv::Mat& output, unsigned char threshold) {
  for (int i = 0; i < input.rows; ++i)
    for (int j = 0; j < input.cols; ++j){
      unsigned char currColor = input.at<cv::Vec3b>(i, j)[0];
      unsigned char newColor;
      newColor = currColor < threshold ? 0 : 255;
      output.at<cv::Vec3b>(i, j) = cv::Vec3b(newColor, newColor, newColor);
    }
}

void OtsuFilter(const cv::Mat & input, cv::Mat & output) {
  Monochrome(input, output);
  unsigned char threshold = OtsuThreshold(output);
  cv::Mat newInput;
  output.copyTo(newInput);
  Binarization(newInput, output, threshold);
}

int fib(int n){
  if(n <= 1)
    return n;
  return fib(n - 1) + fib(n - 2);
}

void histPull(const cv::Mat& sourse, int* hist, unsigned char& count) {
  for (int i = 0; i < 256; ++i)
    hist[i] = 0;
  for (int i = 0; i < sourse.rows; ++i)
    for (int j = 0; j < sourse.cols; ++j)
      hist[sourse.at<cv::Vec3b>(i, j)[0]]++;
  for (int i = 0; i < 256; ++i)
    if (hist[i] != 0) count++;
}

void strconcat(char* str, char* paterncode, char add) {
  int i = 0;
  while (*(paterncode + i) != '\0'){
    *(str + i) = *(paterncode + i);
    ++i;
  }
  str[i] = add;
  str[i + 1] = '\0';
}

void probabilitisOfIntensity(int* hist, double* res, int countOfPixels) {
  for (int i = 0; i < 256; i++)
    res[i] = static_cast<double>(hist[i]) / static_cast<double>(countOfPixels);
}

void HuffmanCompression(const cv::Mat& input, cv::Mat output) {
  // перевод изображение в серое и заполнение гистограммы

  input.copyTo(output);
  Monochrome(input, output);
  int hist[256];
  unsigned char nodes = 0;
  histPull(output, hist, nodes);

  // определение максимальной длины кодовового символа

  double probilitis[256];
  probabilitisOfIntensity(hist, probilitis,
                          output.rows * output.cols);
  unsigned char maxLen = 0;
  double tmp = probilitis[0];
  for (int i = 1; i < 256; ++i)
    if (probilitis[i] < tmp && probilitis[i] > 0) tmp = probilitis[i];
  while (1 / tmp > fib(maxLen)) ++maxLen;
  maxLen -= 3;

  // структура для хранения пикселя

  struct PixelInfo {
    unsigned char intens;
    double prob;
    PixelInfo *left, *right;
    char* code;
    void setSize(unsigned char len) {code = new char[len];}
  };

  // структура для хранения дерева Хаффмана

  struct HuffCode {
    unsigned char intens;
    int arrloc;
    double prob;
  };

  // создание деревьев Хаффмана и инициализация
  int totalCount = 2 * nodes - 1;
  PixelInfo* pixelInfo = new PixelInfo[totalCount];
  for (int i = 0; i < totalCount; ++i) pixelInfo[i].setSize(maxLen);
  HuffCode* huffCode = new HuffCode[totalCount];

  int j = 0;
  double temp;
  int pixelCount = input.rows * input.cols;
  for (int i = 0; i < 256; i++){
    if (hist[i] != 0) {
      huffCode[j].intens = i;
      pixelInfo[j].intens = i;

      huffCode[j].arrloc = j;

      temp = static_cast<double>(hist[i]) / static_cast<double>(pixelCount);
      pixelInfo[j].prob = temp;
      huffCode[j].prob = temp;

      pixelInfo[j].left = NULL;
      pixelInfo[j].right = NULL;

      pixelInfo[j].code[0] = '\0';
      ++j;
    }
  }

  //std::sort(huffCode[0], huffCode[totalCount], [&]()->)
  HuffCode tempHuff;
  for (int i = 0; i < nodes; ++i)
    for (j = i + 1; j < nodes; ++j)
      if (huffCode[i].prob < huffCode[j].prob) {
        tempHuff = huffCode[i];
        huffCode[i] = huffCode[j];
        huffCode[j] = tempHuff;
      }

  double sumprob;
  int sumintens;
  int n = 0, k = 0;
  int nextnode = nodes;
  while (n < nodes - 1) {
    sumprob = huffCode[nodes - n - 1].prob + huffCode[nodes - n - 2].prob;
    sumintens = huffCode[nodes - n - 1].intens + huffCode[nodes - n - 2].intens;

    pixelInfo[nextnode].intens = sumintens;
    pixelInfo[nextnode].prob = sumprob;
    pixelInfo[nextnode].left = &pixelInfo[huffCode[nodes - n - 2].arrloc];
    pixelInfo[nextnode].right = &pixelInfo[huffCode[nodes - n - 1].arrloc];
    pixelInfo[nextnode].code[0] = '\0';

    int i = 0;
    while (sumprob <= huffCode[i].prob) ++i;
    for (k = nodes; k >= 0; --k) {
      if (k == i) {
        huffCode[k].intens = sumintens;
        huffCode[k].prob = sumprob;
        huffCode[k].arrloc = nextnode;
      } else if (k > i)
        huffCode[k] = huffCode[k - 1];
    }
    ++n;
    ++nextnode;
  }

  char left = '0';
  char right = '1';
  int index;
  for (int i = totalCount - 1; i >= nodes; --i) {
    if (pixelInfo[i].left != NULL)
      strconcat(pixelInfo[i].left->code, pixelInfo[i].code, left);
    if (pixelInfo[i].right != NULL)
      strconcat(pixelInfo[i].right->code, pixelInfo[i].code, right);
  }

  for (int i = 0; i < nodes; ++i) {
    std::cout << huffCode[i].intens << " -- " << pixelInfo[i].code;
  }

  delete[] pixelInfo;
  delete[] huffCode;
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

double covFuncion(double& mW1, double& mW2, cv::Mat& a, cv::Mat& b){
    double cov = 0;
    for (int i = 0; i < a.rows; i++)
        for (int j = 0; j < a.cols; j++)
            cov = ((a.at<cv::Vec3b>(i, j)[0] + a.at<cv::Vec3b>(i, j)[1] + a.at<cv::Vec3b>(i, j)[2]) / 3 - mW1) *
            ((b.at<cv::Vec3b>(i, j)[0] + b.at<cv::Vec3b>(i, j)[1] + b.at<cv::Vec3b>(i, j)[2]) / 3 - mW2);
    cov = sqrt(cov);
    return cov;
}

double ssim(double & cE1, double & cE2, double & dis1, double & dis2, double cov){
    
    return (2 * cE1 * cE2 + 0.0001)*(2 * cov + 0.0001)/((pow(cE1,2) + pow(cE2, 2) + 0.0001)*(dis1 + dis2 + 0.0001));
}
