#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat otsu(Mat &input) {
    Mat img = input.clone();
    GaussianBlur(img, img, Size(3, 3), 0.5, 0.5);

    vector<double> hist(256, 0.0);
    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            uchar val = img.at<uchar>(x, y);
            hist[val] += 1.0;
        }
    }

    double tot = img.rows * img.cols;
    for (size_t i = 0; i < hist.size(); i++)
        hist[i] /= tot;

    double gMean = 0.0;
    for (int i = 0; i < 256; i++)
        gMean += i * hist[i];

    double w = 0.0;
    double cMean = 0.0;
    double maxVar = 0.0;
    int bestTH = 0;

    for (int k = 0; k < 256; k++) {
        w += hist[k];

        if (w > 0.0 && w < 1.0) {
            cMean += k * hist[k];
            double var = pow(gMean * w - cMean, 2) / (w * (1.0 - w));
            if (var > maxVar) {
                maxVar = var;
                bestTH = k;
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
