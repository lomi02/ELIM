#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat otsu2k(Mat &input, int blurSize, float blurSigma) {
    Mat img = input.clone();
    GaussianBlur(img, img, Size(blurSize, blurSize), blurSigma);

    vector histogram(256, 0.0);
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            histogram[img.at<uchar>(i, j)]++;

    double totalPixels = img.rows * img.cols;
    for (auto &histValue: histogram) histValue /= totalPixels;

    double globalMean = 0.0;
    for (int i = 0; i < 256; i++)
        globalMean += i * histogram[i];

    double maxVar = 0.0;
    vector optimalTH = {0, 0};

    double w0, w1, w2, m0, m1, m2, variance;

    for (int k1 = 1; k1 < 256 - 2; k1++) {
        w0 = m0 = 0.0;
        for (int i = 0; i <= k1; i++) {
            w0 += histogram[i];
            m0 += i * histogram[i];
        }

        for (int k2 = k1 + 1; k2 < 256 - 1; k2++) {
            w1 = m1 = 0.0;
            for (int i = k1 + 1; i <= k2; i++) {
                w1 += histogram[i];
                m1 += i * histogram[i];
            }

            w2 = 1.0 - w0 - w1;
            m2 = globalMean - m0 - m1;

            variance = w0 * (m0 / w0 - globalMean) * (m0 / w0 - globalMean) +
                       w1 * (m1 / w1 - globalMean) * (m1 / w1 - globalMean) +
                       w2 * (m2 / w2 - globalMean) * (m2 / w2 - globalMean);

            if (variance > maxVar) {
                maxVar = variance;
                optimalTH = {k1, k2};
            }
        }
    }

    Mat out = Mat::zeros(img.size(), CV_8U);
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++) {
            uchar val = img.at<uchar>(i, j);
            if (val >= optimalTH[1])
                out.at<uchar>(i, j) = 255;
            else if (val >= optimalTH[0])
                out.at<uchar>(i, j) = 127;
        }

    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/fiore.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    int blurSize = 3;
    float blurSigma = 0.5;
    Mat dst = otsu2k(src, blurSize, blurSigma);

    imshow("Otsu2k", dst);
    waitKey(0);

    return 0;
}
