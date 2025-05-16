#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat hough_lines(Mat & input, int houghTH, int cannyTHL, int cannyTHH, int blurSize, float blurSigma) {
    Mat img = input.clone();
    GaussianBlur(img, img, Size(blurSize, blurSize), blurSigma);
    Canny(img, img, cannyTHL, cannyTHH);

    int diagonalLength = cvRound(hypot(img.rows, img.cols));
    int maxTheta = 180;
    Mat votes = Mat::zeros(diagonalLength * 2, maxTheta, CV_8U);

    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            if (img.at<uchar>(Point(x, y)) == 255)
                for (int theta = 0; theta < maxTheta; ++theta) {
                    int rho = cvRound(x * cos(theta) + y * sin(theta));
                    int rhoIndex = rho + diagonalLength;
                    votes.at<uchar>(rhoIndex, theta)++;
                }

    Mat lineImg = input.clone();
    int lineOffset = diagonalLength * 2;

    for (int rhoIndex = 0; rhoIndex < votes.rows; ++rhoIndex)
        for (int theta = 0; theta < votes.cols; ++theta)
            if (votes.at<uchar>(rhoIndex, theta) > houghTH) {
                int rho = rhoIndex - diagonalLength;
                int x0 = cvRound(rho * cos(theta));
                int y0 = cvRound(rho * sin(theta));

                Point point1;
                point1.x = cvRound(x0 + lineOffset * -sin(theta));
                point1.y = cvRound(y0 + lineOffset * cos(theta));

                Point point2;
                point2.x = cvRound(x0 - lineOffset * -sin(theta));
                point2.y = cvRound(y0 - lineOffset * cos(theta));

                line(lineImg, point1, point2, Scalar(0), 2, 0);
            }
    return lineImg;
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
