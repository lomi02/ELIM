#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat otsu(Mat &input) {
    Mat img = input.clone();
    GaussianBlur(img, img, Size(3, 3), 0.5, 0.5);

    vector<double> hist(256, 0.0);
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            hist[img.at<uchar>(x, y)]++;

    double tot = img.rows * img.cols;
    for (int i = 0; i < 256; i++)
        hist[i] /= tot;

    double gMean = 0.0;
    for (int i = 0; i < 256; i++)
        gMean += i * hist[i];

    double w = 0.0, mean = 0.0, maxVar = 0.0;
    int bestTH = 0;

    for (int i = 0; i < 256; i++) {
        w += hist[i];

        if (w > 0.0 && w < 1.0) {
            mean += i * hist[i];

            double var = pow(gMean * w - mean, 2) / (w * (1.0 - w));
            if (var > maxVar) {
                maxVar = var;
                bestTH = i;
            }
        }
    }

    Mat out;
    threshold(img, out, bestTH, 255, THRESH_BINARY);
    return out;
}

int main() {
    Mat src = imread("../immagini/fiore.png", IMREAD_GRAYSCALE);

    Mat dst = otsu(src);

    imshow("Otsu", dst);
    waitKey(0);

    return 0;
}
