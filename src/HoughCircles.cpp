#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat hough_circles(Mat &input, int houghTH, int Rmin, int Rmax) {
    Mat img = input.clone();

    GaussianBlur(img, img, Size(3, 3), 1, 1);
    Canny(img, img, 100, 250);

    vector<vector<vector<int>>> votes(img.rows, vector<vector<int>>(img.cols, vector<int>(Rmax - Rmin + 1, 0)));

    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            if (img.at<uchar>(x, y) == 255)
                for (int r = Rmin; r < Rmax; r++)
                    for (int thetaDeg = 0; thetaDeg < 360; thetaDeg++) {
                        double thetaRad = thetaDeg * CV_PI / 180.0;

                        int a = y - r * cos(thetaRad);
                        int b = x - r * sin(thetaRad);

                        if (a >= 0 && a < img.cols && b >= 0 && b < img.rows)
                            votes[b][a][r - Rmin]++;
                    }

    Mat out = input.clone();
    for (int b = 0; b < img.rows; b++)
        for (int a = 0; a < img.cols; a++)
            for (int r = Rmin; r < Rmax; r++)
                if (votes[b][a][r - Rmin] > houghTH)
                    circle(out, Point(a, b), r, Scalar(0), 1);

    return out;
}

int main() {
    Mat src = imread("../immagini/monete.png", IMREAD_GRAYSCALE);

    int houghTH = 175;
    int Rmin = 20;
    int Rmax = 70;

    Mat dst = hough_circles(src, houghTH, Rmin, Rmax);

    imshow("Hough Circles", dst);
    waitKey(0);

    return 0;
}
