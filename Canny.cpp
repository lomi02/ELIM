#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void hysteresisThreshold(Mat &src, Mat &dst, int lth, int hth) {
    dst = Mat::zeros(src.size(), CV_8U);
    for (int i = 1; i < src.rows; i++)
        for (int j = 1; j < src.cols; j++)
            if (src.at<uchar>(i, j) > hth) {
                dst.at<uchar>(i, j) = 255;
                for (int k = -1; k <= 1; k++)
                    for (int l = -1; l <= 1; l++)
                        if (src.at<uchar>(i + k, j + l) >= lth && src.at<uchar>(i + k, j + l) <= hth)
                            dst.at<uchar>(i + k, j + l) = 255;
            } else
                dst.at<uchar>(i, j) = 0;
}

void nonMaximaSuppression(Mat &mag, Mat &orientations, Mat &nms) {
    nms = Mat::zeros(mag.size(), CV_8U);
    for (int i = 1; i < mag.rows - 1; i++)
        for (int j = 1; j < mag.cols - 1; j++) {
            float angle = orientations.at<float>(i, j);
            if (angle > 180)
                angle -= 360;
            int dx = 0, dy = 0;

            // 0°(+-22.5) || 180°(+-22.5)
            if ((-22.5 < angle && angle <= 22.5) || (157.5 < angle && angle <= -157.5))
                dx = 1;

            // 45°(+-22.5) || -135°(+-22.5)
            if ((22.5 < angle && angle <= 67.5) || (-157.5 < angle && angle <= -112.5)) {
                dx = 1;
                dy = 1;
            }

            // 90°(+-22.5) || -90°(+-22.5)
            if ((67.5 < angle && angle <= 112.5) || (-112.5 < angle && angle <= -67.5))
                dy = 1;

            // 135°(+-22.5) || -45°(+-22.5)
            if ((112.5 < angle && angle <= 157.5) || (-67.5 < angle && angle <= -22.5)) {
                dx = 1;
                dy = -1;
            }

            if (mag.at<uchar>(i, j) >= mag.at<uchar>(i + dy, j + dx) && mag.at<uchar>(i, j) >= mag.at<uchar>(i - dy, j - dx))
                nms.at<uchar>(i, j) = mag.at<uchar>(i, j);
        }
}

void canny(Mat &src, Mat &dst, int lth, int hth) {
    // 1. Gaussian Blur
    Mat gauss;
    GaussianBlur(src, gauss, Size(3, 3), 0, 0);

    // 2. Gradient Calculation
    Mat Dx, Dy, mag, orientations;
    Sobel(gauss, Dx, CV_32F, 1, 0, 3);
    Sobel(gauss, Dy, CV_32F, 0, 1, 3);
    magnitude(Dx, Dy, mag);
    normalize(mag, mag, 0, 255, NORM_MINMAX, CV_8U);
    phase(Dx, Dy, orientations, true);

    // 3. Non-Maximum Suppression
    Mat nms;
    nonMaximaSuppression(mag, orientations, nms);

    // 4. Hysteresis Thresholding
    hysteresisThreshold(nms, dst, lth, hth);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/fiore.png";
    Mat dst, src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    int lth = 10, hth = 50;
    canny(src, dst, lth, hth);

    imshow("Canny", dst);
    waitKey(0);
    return 0;
}
