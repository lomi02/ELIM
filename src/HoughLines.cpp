#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat hough_lines(Mat & input, int houghTH, int cannyTHL, int cannyTHH, int blurSize, float blurSigma) {
    Mat img = input.clone();
    GaussianBlur(img, img, Size(blurSize, blurSize), blurSigma);
    Canny(img, img, cannyTHL, cannyTHH);

    int diagonalLength = cvRound(hypot(img.rows, img.cols));
    Mat votes = Mat::zeros(diagonalLength * 2, 180, CV_8U);

    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            if (img.at<uchar>(Point(x, y)) == 255)
                for (int thetaDeg = 0; thetaDeg < 180; ++thetaDeg) {
                    double theta = thetaDeg * CV_PI / 180.0;
                    int rho = cvRound(x * cos(theta) + y * sin(theta));
                    int rhoIndex = rho + diagonalLength;
                    votes.at<uchar>(rhoIndex, thetaDeg)++;
                }

    Mat out = input.clone();
    int lineLength = max(img.rows, img.cols);

    for (int rhoIndex = 0; rhoIndex < votes.rows; ++rhoIndex)
        for (int thetaDeg = 0; thetaDeg < votes.cols; ++thetaDeg)
            if (votes.at<uchar>(rhoIndex, thetaDeg) > houghTH) {
                double theta = thetaDeg * CV_PI / 180.0;
                int rho = rhoIndex - diagonalLength;

                double a = cos(theta), b = sin(theta);
                double x0 = a * rho, y0 = b * rho;

                Point point1, point2;
                point1.x = cvRound(x0 + lineLength * -b);
                point1.y = cvRound(y0 + lineLength * a);
                point2.x = cvRound(x0 - lineLength * -b);
                point2.y = cvRound(y0 - lineLength * a);

                line(out, point1, point2, Scalar(0), 2);
            }
    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/strada.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    int houghTH = 150;
    int cannyTHL  = 40;
    int cannyTHH  = 80;
    int blurSize  = 1;
    float blurSigma  = 0.0;
    Mat dst = hough_lines(src, houghTH, cannyTHL, cannyTHH, blurSize, blurSigma);

    imshow("HoughLines", dst);
    waitKey(0);
    return 0;
}
