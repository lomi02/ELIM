#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

Mat otsu2k(Mat &input) {
    Mat img = input.clone();
    GaussianBlur(img, img, Size(3, 3), 1);

    vector hist(256, 0.0);
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            hist[img.at<uchar>(x, y)]++;

    double totalPixels = img.rows * img.cols;
    for (double &val: hist)
        val /= totalPixels;

    vector<double> cumProb(256, 0.0), cumMean(256, 0.0);
    cumProb[0] = hist[0];
    cumMean[0] = 0.0;
    for (int i = 1; i < 256; ++i) {
        cumProb[i] = cumProb[i - 1] + hist[i];
        cumMean[i] = cumMean[i - 1] + i * hist[i];
    }

    double globalMean = cumMean[255];
    double maxVar = 0.0;
    vector th(2, 0);

    for (int k1 = 1; k1 < 254; ++k1)
        for (int k2 = k1 + 1; k2 < 255; ++k2) {
            double w0 = cumProb[k1];
            double w1 = cumProb[k2] - cumProb[k1];
            double w2 = 1.0 - cumProb[k2];

            double m0 = w0 > 0 ? cumMean[k1] / w0 : 0.0;
            double m1 = w1 > 0 ? (cumMean[k2] - cumMean[k1]) / w1 : 0.0;
            double m2 = w2 > 0 ? (cumMean[255] - cumMean[k2]) / w2 : 0.0;

            double var = w0 * (m0 - globalMean) * (m0 - globalMean) +
                         w1 * (m1 - globalMean) * (m1 - globalMean) +
                         w2 * (m2 - globalMean) * (m2 - globalMean);

            if (var > maxVar) {
                maxVar = var;
                th[0] = k1;
                th[1] = k2;
            }
        }

    Mat out = Mat::zeros(img.size(), CV_8U);
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++) {
            uchar pixel = img.at<uchar>(x, y);
            if (pixel >= th[1]) out.at<uchar>(x, y) = 255;
            else if (pixel >= th[0]) out.at<uchar>(x, y) = 127;
        }

    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/fiore.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    Mat dst = otsu2k(src);

    imshow("Otsu2k", dst);
    waitKey(0);

    return 0;
}
