#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat hough_circles(Mat &input, int houghTH, int Rmin, int Rmax) {
    Mat img = input.clone();

    GaussianBlur(img, img, Size(5, 5), 0.5, 0.5);
    Canny(img, img, 100, 250);

    vector votes(img.rows, vector(img.cols, vector(Rmax - Rmin + 1, 0)));

    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            if (img.at<uchar>(x, y) == 255)
                for (int r = Rmin; r < Rmax; r++)
                    for (int theta = 0; theta < 360; theta++) {

                        int a = y - r * cos(theta * CV_PI / 180);
                        int b = x - r * sin(theta * CV_PI / 180);

                        if (a >= 0 && a < img.rows && b >= 0 && b < img.cols)
                            votes[a][b][r - Rmin]++;
                    }

    Mat out = input.clone();
    for (int a = 0; a < img.rows; a++)
        for (int b = 0; b < img.cols; b++)
            for (int r = Rmin; r < Rmax; r++)
                if (votes[a][b][r - Rmin] > houghTH)
                    circle(out, Point(a, b), r, Scalar(0));

    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/monete.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    int houghTH = 175;
    int Rmin = 20;
    int Rmax = 70;

    Mat dst = hough_circles(src, houghTH, Rmin, Rmax);

    imshow("Hough Circles", dst);
    waitKey(0);

    return 0;
}
