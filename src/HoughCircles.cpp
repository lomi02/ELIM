#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat hough_circles(Mat &input, int houghTH, int radiusMin, int radiusMax, int cannyTHL, int cannyTHH, int blurSize = 3, float blurSigma = 0.5) {
    Mat img = input.clone();
    GaussianBlur(img, img, Size(blurSize, blurSize), blurSigma);
    Canny(img, img, cannyTHL, cannyTHH);

    int radiusOffset = radiusMax - radiusMin + 1;
    int sizes[] = {img.cols, img.rows, radiusOffset};
    auto votes = Mat(3, sizes, CV_8U, Scalar(0));

    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            if (img.at<uchar>(Point(x, y)) == 255)
                for (int radius = radiusMin; radius < radiusMax; ++radius)
                    for (int thetaDegrees = 0; thetaDegrees < 360; ++thetaDegrees) {
                        double thetaRadians = thetaDegrees * CV_PI / 180;
                        int alpha = cvRound(x - radius * cos(thetaRadians));
                        int beta = cvRound(y - radius * sin(thetaRadians));
                        if (alpha >= 0 and alpha < img.cols and beta >= 0 and beta < img.rows)
                            votes.at<uchar>(alpha, beta, radius - radiusMin)++;
                    }

    Mat out = input.clone();

    for (int radius = radiusMin; radius < radiusMax; ++radius)
        for (int alpha = 0; alpha < img.cols; ++alpha)
            for (int beta = 0; beta < img.rows; ++beta)
                if (votes.at<uchar>(alpha, beta, radius - radiusMin) > houghTH)
                    circle(out, Point(alpha, beta), radius, Scalar(0), 2, 8);

    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/monete.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    int houghTH = 190;
    int radMin = 20;
    int radMax = 70;
    int cannyTHL = 40;
    int cannyTHH = 80;
    int blurSize = 1;
    float blurSigma  = 0.0;
    Mat dst = hough_circles(src, houghTH, radMin, radMax, cannyTHL, cannyTHH, blurSize, blurSigma);

    imshow("Hough Circles", dst);
    waitKey(0);

    return 0;
}
