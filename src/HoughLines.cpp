#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat hough_lines(Mat &input, int houghTH) {
    Mat img = input.clone();

    GaussianBlur(img, img, Size(5, 5), 0.5, 0.5);
    Canny(img, img, 50, 150);

    int diag = cvRound(hypot(img.rows, img.cols));
    Mat votes = Mat::zeros(diag * 2, 180, CV_8U);

    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            if (img.at<uchar>(x, y) == 255)
                for (int thetaDeg = 0; thetaDeg < 180; thetaDeg++) {
                    double thetaRad = thetaDeg * CV_PI / 180.0;
                    int rho = cvRound(x * sin(thetaRad) + y * cos(thetaRad)) + diag;

                    votes.at<uchar>(rho, thetaDeg)++;
                }

    Mat out = input.clone();
    int lineLength = max(img.rows, img.cols);
    for (int rho = 0; rho < votes.rows; rho++)
        for (int thetaDeg = 0; thetaDeg < votes.cols; thetaDeg++)
            if (votes.at<uchar>(rho, thetaDeg) > houghTH) {
                double thetaRad = thetaDeg * CV_PI / 180.0;
                double a = cos(thetaRad), b = sin(thetaRad);
                double x0 = a * (rho - diag);
                double y0 = b * (rho - diag);

                Point p1(cvRound(x0 - lineLength * b), cvRound(y0 + lineLength * a));
                Point p2(cvRound(x0 + lineLength * b), cvRound(y0 - lineLength * a));

                line(out, p1, p2, Scalar(0), 2);
            }

    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/strada.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    int houghTH = 150;
    Mat dst = hough_lines(src, houghTH);

    imshow("HoughLines", dst);
    waitKey(0);
    return 0;
}
