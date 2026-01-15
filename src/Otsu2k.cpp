#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat otsu2k(Mat &input) {
    Mat img = input.clone();
    GaussianBlur(img, img, Size(3, 3), 0.5, 0.5);

    vector<double> hist(256, 0.0);
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            hist[img.at<uchar>(x, y)]++;

    double gMean = 0.0;
    for (int i = 0; i < 256; i++)
        gMean += i * hist[i];

    double maxVar = 0.0;
    int bestTH1 = 0, bestTH2 = 0;

    for (int t1 = 0; t1 < 255; t1++)
        for (int t2 = t1 + 1; t2 < 256; t2++) {
            double w0 = 0.0, w1 = 0.0, w2 = 0.0;
            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0;

            for (int i = 0; i <= t1; i++) {
                w0 += hist[i];
                sum0 += i * hist[i];
            }
            for (int i = t1 + 1; i <= t2; i++) {
                w1 += hist[i];
                sum1 += i * hist[i];
            }
            for (int i = t2 + 1; i < 256; i++) {
                w2 += hist[i];
                sum2 += i * hist[i];
            }

            if (w0 > 0 && w1 > 0 && w2 > 0) {
                double mean0 = sum0 / w0;
                double mean1 = sum1 / w1;
                double mean2 = sum2 / w2;

                double var = w0 * pow(mean0 - gMean, 2) +
                             w1 * pow(mean1 - gMean, 2) +
                             w2 * pow(mean2 - gMean, 2);

                if (var > maxVar) {
                    maxVar = var;
                    bestTH1 = t1;
                    bestTH2 = t2;
                }
            }
        }

    Mat out = img.clone();
    for (int x = 0; x < out.rows; x++)
        for (int y = 0; y < out.cols; y++) {
            uchar val = out.at<uchar>(x, y);
            if (val <= bestTH1)
                out.at<uchar>(x, y) = 0;
            else if (val <= bestTH2)
                out.at<uchar>(x, y) = 127;
            else
                out.at<uchar>(x, y) = 255;
        }

    return out;
}

int main() {
    Mat src = imread("../immagini/fiore.png", IMREAD_GRAYSCALE);

    Mat dst = otsu2k(src);

    imshow("Otsu2k", dst);
    waitKey(0);

    return 0;
}
