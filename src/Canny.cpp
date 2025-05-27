#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat canny(Mat &input, int cannyLTH, int cannyHTH) {
    Mat img = input.clone();
    GaussianBlur(img, img, Size(3, 3), 0.5, 0.5);

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

            uchar curr = magnitude.at<uchar>(x, y);
            uchar pixel1, pixel2;

            if (angle < 45) {
                pixel1 = magnitude.at<uchar>(x + 1, y);
                pixel2 = magnitude.at<uchar>(x - 1, y);
            }

            else if (angle < 90) {
                pixel1 = magnitude.at<uchar>(x + 1, y - 1);
                pixel2 = magnitude.at<uchar>(x - 1, y + 1);
            }

            else if (angle < 135) {
                pixel1 = magnitude.at<uchar>(x, y + 1);
                pixel2 = magnitude.at<uchar>(x, y - 1);
            }

            else {
                pixel1 = magnitude.at<uchar>(x + 1, y + 1);
                pixel2 = magnitude.at<uchar>(x - 1, y - 1);
            }

            if (curr >= pixel1 && curr >= pixel2)
                NMS.at<uchar>(x, y) = curr;
        }

    Mat out = Mat::zeros(NMS.size(), CV_8U);
    for (int x = 0; x < NMS.rows; x++)
        for (int y = 0; y < NMS.cols; y++) {
            uchar val = NMS.at<uchar>(x, y);
            if (val > cannyLTH)
                out.at<uchar>(x, y) = 255;

            for (int nx = -1; nx <= 1; nx++)
                for (int ny = -1; ny <= 1; ny++)
                    if (NMS.at<uchar>(x + nx, y + ny) >= cannyLTH && NMS.at<uchar>(x + nx, y + ny) <= cannyHTH)
                        out.at<uchar>(x + nx, y + ny) = 255;
                    else
                        out.at<uchar>(x, y) = 0;
        }

    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/fiore.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    int cannyLTH = 20;
    int cannyHTH = 150;

    Mat dst = canny(src, cannyLTH, cannyHTH);

    imshow("Canny", dst);
    waitKey(0);
    return 0;
}
