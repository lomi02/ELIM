#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat canny(Mat &input, int cannyLTH, int cannyHTH) {
    Mat img = input.clone();
    GaussianBlur(img, img, Size(3, 3), 1, 1);

    Mat Dx, Dy;
    Sobel(img, Dx, CV_32F, 1, 0);
    Sobel(img, Dy, CV_32F, 0, 1);

    Mat Dx2, Dy2, magnitude;
    multiply(Dx, Dx, Dx2);
    multiply(Dy, Dy, Dy2);
    sqrt(Dx2 + Dy2, magnitude);

    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8U);
    Mat phase;
    cv::phase(Dx, Dy, phase);

    Mat NMS = Mat::zeros(magnitude.size(), CV_8U);
    for (int x = 1; x < magnitude.rows; x++)
        for (int y = 1; y < magnitude.cols; y++) {
            float angle = phase.at<float>(x, y);
            angle = fmod(angle + 22.5, 180);

            uchar curr = magnitude.at<uchar>(x, y);
            uchar pixel1, pixel2;

            if (angle < 45) {
                pixel1 = magnitude.at<uchar>(x + 1, y);
                pixel2 = magnitude.at<uchar>(x - 1, y);
            } else if (angle < 90) {
                pixel1 = magnitude.at<uchar>(x + 1, y - 1);
                pixel2 = magnitude.at<uchar>(x - 1, y + 1);
            } else if (angle < 135) {
                pixel1 = magnitude.at<uchar>(x, y + 1);
                pixel2 = magnitude.at<uchar>(x, y - 1);
            } else {
                pixel1 = magnitude.at<uchar>(x + 1, y + 1);
                pixel2 = magnitude.at<uchar>(x - 1, y - 1);
            }

            if (curr >= pixel1 && curr >= pixel2)
                NMS.at<uchar>(x, y) = curr;
        }

    Mat out = Mat::zeros(NMS.size(), CV_8U);
    for (int x = 1; x < NMS.rows; x++)
        for (int y = 1; y < NMS.cols; y++)
            if (NMS.at<uchar>(x, y) > cannyLTH && NMS.at<uchar>(x, y) < cannyHTH)
                out.at<uchar>(x, y) = 255;

    return out;
}

int main() {
    Mat src = imread("../immagini/fiore.png", IMREAD_GRAYSCALE);

    int cannyLTH = 20;
    int cannyHTH = 150;

    Mat dst = canny(src, cannyLTH, cannyHTH);

    imshow("Canny", dst);
    waitKey(0);
    return 0;
}
