#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat harris(Mat & input, float k, int sobelSize, int threshTH = 70, int blurSize = 3, float blurSigma = 0.5) {
    Mat img = input.clone();

    Mat Ix, Iy;
    Sobel(img, Ix, CV_32F, 1, 0, sobelSize);
    Sobel(img, Iy, CV_32F, 0, 1, sobelSize);

    Mat IxIy;
    multiply(Ix, Iy, IxIy);

    pow(Ix, 2, Ix);
    pow(Iy, 2, Iy);

    GaussianBlur(Ix, Ix, Size(blurSize, blurSize), blurSigma, blurSigma);
    GaussianBlur(Iy, Iy, Size(blurSize, blurSize), blurSigma, blurSigma);
    GaussianBlur(IxIy, IxIy, Size(blurSize, blurSize), blurSigma, blurSigma);

    Mat detM;
    multiply(Ix, Iy, detM);
    Mat IxIy_squared;
    pow(IxIy, 2, IxIy_squared);
    detM -= IxIy_squared;

    Mat traceM;
    pow(Ix + Iy, 2, traceM);

    Mat harrisResponse = detM - k * traceM;
    normalize(harrisResponse, harrisResponse, 0, 255, NORM_MINMAX, CV_8U);
    threshold(harrisResponse, harrisResponse, threshTH, 255, THRESH_BINARY);

    Mat out = input.clone();
    for (int y = 0; y < harrisResponse.rows; ++y)
        for (int x = 0; x < harrisResponse.cols; ++x)
            if (harrisResponse.at<uchar>(Point(x, y)) > 0)
                circle(out, Point(x, y), 3, Scalar(0), 1, 8, 0);

    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/foglia.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    float k = 0.05;
    int sobelSize = 3;
    int blurSize = 3;
    float blurSigma = 2.0;
    int threshTH = 115;
    Mat dst = harris(src, k, sobelSize, threshTH, blurSize, blurSigma);

    imshow("Harris", dst);
    waitKey(0);

    return 0;
}
