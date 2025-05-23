#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat canny(Mat &input, int cannyTHL, int cannyTHH, int blurSize, float blurSigma) {
    Mat img = input.clone();
    GaussianBlur(img, img, Size(blurSize, blurSize), blurSigma, blurSigma);

    Mat x_gradient, y_gradient;
    Sobel(img, x_gradient, CV_32F, 1, 0);
    Sobel(img, y_gradient, CV_32F, 0, 1);

    Mat x_gradient2, y_gradient2, magnitude;
    pow(x_gradient, 2, x_gradient2);
    pow(y_gradient, 2, y_gradient2);
    sqrt(x_gradient2 + y_gradient2, magnitude);
    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8U);

    Mat phase;
    cv::phase(x_gradient, y_gradient, phase);

    Mat NMS = Mat::zeros(magnitude.size(), CV_8U);
    for (int x = 1; x < magnitude.rows - 1; x++)
        for (int y = 1; y < magnitude.cols - 1; y++) {
            float angle = phase.at<float>(x, y);
            angle = fmod(angle + 22.5, 180);

            uchar pixel1, pixel2;
            uchar curr = magnitude.at<uchar>(x, y);

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

    Mat edges = Mat::zeros(NMS.size(), CV_8U);
    for (int x = 0; x < NMS.cols; x++)
        for (int y = 0; y < NMS.rows; y++) {
            uchar val = NMS.at<uchar>(y, x);
            if (val >= cannyTHH) {
                edges.at<uchar>(y, x) = 255;
                for (int dx = -1; dx <= 1; dx++)
                    for (int dy = -1; dy <= 1; dy++) {
                        int nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < NMS.cols && ny >= 0 && ny < NMS.rows &&
                            NMS.at<uchar>(ny, nx) >= cannyTHL) {
                            edges.at<uchar>(ny, nx) = 255;
                        }
                    }
            }
        }

    return edges;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/fiore.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    int cannyTHL = 5;
    int cannyTHH = 20;
    int blurSize = 3;
    float blurSigma = 0.5;

    Mat dst = canny(src, cannyTHL, cannyTHH, blurSize, blurSigma);

    imshow("Canny", dst);
    waitKey(0);
    return 0;
}
