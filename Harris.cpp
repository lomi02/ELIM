#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void harris(Mat &src, Mat &dst, int th, float k) {
    dst = src.clone();

    // 1: Sobel
    Mat Dx, Dy;
    Sobel(src, Dx, CV_32F, 1, 0, 3);
    Sobel(src, Dy, CV_32F, 0, 1, 3);

    // 2: DxDy
    Mat Dx2 = Dx.mul(Dx);
    Mat Dy2 = Dy.mul(Dy);
    Mat DxDy = Dx.mul(Dy);

    // 3: Gaussian Blur
    Mat C00, C11, C01;
    GaussianBlur(Dx2, C00, Size(7, 7), 2.0);
    GaussianBlur(Dy2, C11, Size(7, 7), 2.0);
    GaussianBlur(DxDy, C01, Size(7, 7), 2.0);

    // 4: Harris
    Mat det = C00.mul(C11) - C01.mul(C01);  // (C00 * C11) - (C01 * C10)
    Mat trace = C00 + C11;
    Mat R = det - k * trace.mul(trace);     // det - k * trace^2

    // 5: Normalize and circle corners
    normalize(R, R, 0, 255, NORM_MINMAX, CV_32F);

    // &: Circle Corners
    for (int i = 0; i < R.rows; i++)
        for (int j = 0; j < R.cols; j++)
            if ((int) R.at<float>(i, j) > th)
                circle(dst, Point(j, i), 7, Scalar(0), 2);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/foglia.png";
    Mat dst, src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    harris(src, dst, 50, 0.03F);

    imshow("Sorgente", src);
    imshow("Harris", dst);
    waitKey(0);

    return 0;
}
