#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat harris(Mat &input, float k, int threshTH) {
    Mat img = input.clone();

    Mat Dx, Dy;
    Sobel(img, Dx, CV_32F, 1, 0);
    Sobel(img, Dy, CV_32F, 0, 1);

    Mat Dx2, Dy2, DxDy;
    multiply(Dx, Dx, Dx2);
    multiply(Dy, Dy, Dy2);
    multiply(Dx, Dy, DxDy);

    GaussianBlur(Dx2, Dx2, Size(3, 3), 0.5, 0.5);
    GaussianBlur(Dy2, Dy2, Size(3, 3), 0.5, 0.5);
    GaussianBlur(DxDy, DxDy, Size(3, 3), 0.5, 0.5);

    Mat det = Dx2.mul(Dy2) - DxDy.mul(DxDy);
    Mat trace = Dx2 + Dy2;
    Mat R = det - k * trace.mul(trace);

    normalize(R, R, 0, 255, NORM_MINMAX, CV_8U);
    threshold(R, R, threshTH, 255, THRESH_BINARY);

    Mat out = input.clone();
    for (int x = 0; x < R.rows; x++)
        for (int y = 0; y < R.cols; y++)
            if (R.at<uchar>(x, y) > 0)
                circle(out, Point(y, x), 3 , Scalar(0));

    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/foglia.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    float k = 0.017;
    int threshTH = 117;
    Mat dst = harris(src, k, threshTH);

    imshow("Harris", dst);
    waitKey(0);

    return 0;
}
