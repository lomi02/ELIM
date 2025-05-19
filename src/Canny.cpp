#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat canny(Mat & input, int cannyTHL, int cannyTHH, int blurSize = 3, float blurSigma = 0.5) {
    Mat img = input.clone();
    GaussianBlur(img, img, Size(blurSize, blurSize), blurSigma, blurSigma);

    Mat x_gradient, y_gradient;
    Sobel(img, x_gradient, CV_32F, 1, 0);
    Sobel(img, y_gradient, CV_32F, 0, 1);

    Mat magnitude = abs(x_gradient) + abs(y_gradient);
    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8U);

    Mat phase;
    cv::phase(x_gradient, y_gradient, phase);

    Mat NMS = magnitude.clone();
    uchar pixel1, pixel2;

    for (int y = 1; y < magnitude.rows - 1; ++y) {
        for (int x = 1; x < magnitude.cols - 1; ++x) {
            float angle = phase.at<float>(Point(x, y));

            if ((angle >= 360-22.5 && angle <= 22.5)
            || (angle >= 360-22.5+180 && angle <= 22.5+180)) {
                pixel1 = magnitude.at<uchar>(Point(x + 1, y));
                pixel2 = magnitude.at<uchar>(Point(x - 1, y));
            }
            else if ((angle >= 22.5 && angle <= 22.5+45)
            || (angle >= 22.5+180 && angle <= 22.5+45+180)) {
                pixel1 = magnitude.at<uchar>(Point(x - 1, y + 1));
                pixel2 = magnitude.at<uchar>(Point(x + 1, y - 1));
            }
            else if ((angle >= 22.5+45 && angle <= 22.5+90)
            || (angle >= 22.5+45+180 && angle <= 22.5+90+180)) {
                pixel1 = magnitude.at<uchar>(Point(x, y + 1));
                pixel2 = magnitude.at<uchar>(Point(x, y - 1));
            }
            else {
                pixel1 = magnitude.at<uchar>(Point(x + 1, y - 1));
                pixel2 = magnitude.at<uchar>(Point(x - 1, y + 1));
            }

            uchar currentMagnitude = magnitude.at<uchar>(Point(x, y));
            if (currentMagnitude < pixel1 || currentMagnitude < pixel2)
                NMS.at<uchar>(Point(x, y)) = 0;
        }
    }

    Mat edgesImg = Mat::zeros(NMS.rows, NMS.cols, NMS.type());
    for (int y = 0; y < NMS.rows; ++y)
        for (int x = 0; x < NMS.cols; ++x)
            if (NMS.at<uchar>(Point(x, y)) > cannyTHH) {
                edgesImg.at<uchar>(Point(x, y)) = 255;

                Rect roi(x - 1, y - 1, 3, 3);
                for (int roi_y = roi.y; roi_y < roi.y + roi.height; ++roi_y)
                    for (int roi_x = roi.x; roi_x < roi.x + roi.width; ++roi_x)
                        if (NMS.at<uchar>(Point(roi_x, roi_y)) > cannyTHL
                            && NMS.at<uchar>(Point(roi_x, roi_y)) < cannyTHH)
                            edgesImg.at<uchar>(Point(roi_x, roi_y)) = 255;
            }

    return edgesImg;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/fiore.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    int cannyTHL = 5;
    int cannyTHH = 20;
    int blurSize = 21;

    Mat dst = canny(src, cannyTHL, cannyTHH, blurSize);

    imshow("Canny", dst);
    waitKey(0);
    return 0;
}
