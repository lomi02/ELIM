#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat otsu2k(Mat& input, int blurSize = 3, float blurSigma = 0.5) {
    Mat img = input.clone();

    vector histogram(256, 0.0);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            histogram.at(img.at<uchar>(Point(x, y)))++;

    int numberOfPixels = img.rows * img.cols;
    for (double & bin : histogram)
        bin /= numberOfPixels;

    double globalCumulativeMean = 0.0;
    for (int i = 0; i < histogram.size(); ++i)
        globalCumulativeMean += i * histogram.at(i);

    vector probabilities(3, 0.0);
    vector cumulativeMeans(3, 0.0);
    double maxVariance = 0.0;
    vector optimalTH(2, 0);

    for (int k1 = 0; k1 < histogram.size() - 2; ++k1) {
        probabilities.at(0) += histogram.at(k1);
        cumulativeMeans.at(0) += k1 * histogram.at(k1);

        for (int k2 = k1 + 1; k2 < histogram.size() - 1; ++k2) {
            probabilities.at(1) += histogram.at(k2);
            cumulativeMeans.at(1) += k2 * histogram.at(k2);

            for (int k = k2 + 1; k < histogram.size(); ++k) {
                probabilities.at(2) += histogram.at(k);
                cumulativeMeans.at(2) += k * histogram.at(k);

                double betweenClassesVariance = 0.0;
                for (int i = 0; i < 3; ++i) {
                    if (probabilities.at(i) > 0) {
                        double currentCumulativeMean = cumulativeMeans.at(i) / probabilities.at(i);
                        betweenClassesVariance += probabilities.at(i) * pow(currentCumulativeMean - globalCumulativeMean, 2);
                    }
                }

                if (betweenClassesVariance > maxVariance) {
                    maxVariance = betweenClassesVariance;
                    optimalTH.at(0) = k1;
                    optimalTH.at(1) = k2;
                }
            }
            probabilities.at(2) = 0.0;
            cumulativeMeans.at(2) = 0.0;
        }
        probabilities.at(1) = 0.0;
        cumulativeMeans.at(1) = 0.0;
    }

    GaussianBlur(img, img, Size(blurSize, blurSize), blurSigma, blurSigma);

    Mat thresholdedImg = Mat::zeros(img.rows, img.cols, CV_8U);
    for (int x = 0; x < img.rows; ++x) {
        for (int y = 0; y < img.cols; ++y) {
            uchar pixel = img.at<uchar>(x, y);
            if (pixel >= optimalTH.at(1))
                thresholdedImg.at<uchar>(x, y) = 255;
            else if (pixel >= optimalTH.at(0))
                thresholdedImg.at<uchar>(x, y) = 128;
        }
    }
    return thresholdedImg;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/splash.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    Mat dst = otsu2k(src);
    imshow("Otsu2k", dst);
    waitKey(0);

    return 0;
}
