#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

Mat otsu2k(Mat &input) {
    Mat img = input.clone();
    GaussianBlur(img, img, Size(3, 3), 0.5, 0.5);

    vector hist(256, 0.0);
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            hist[img.at<uchar>(x, y)]++;

    double tot = img.rows * img.cols;
    for (double &val: hist)
        val /= tot;

    double gMean = 0.0;
    for (int i = 0; i < 256; i++)
        gMean += i * hist[i];

    double maxVar = 0.0;
    int bestTH1 = 0, bestTH2 = 0;

    for (int i = 0; i < 254; i++) {
        double w0 = 0, m0 = 0;

        for (int k = 0; k <= i; k++) {
            w0 += hist[k];
            m0 += k * hist[k];
        }

        for (int j = i + 1; j < 255; j++) {
            double w1 = 0, m1 = 0;

            for (int k = i + 1; k <= j; k++) {
                w1 += hist[k];
                m1 += k * hist[k];
            }

            double w2 = 1.0 - w0 - w1;
            double m2 = gMean - m0 - m1;

            if (w0 > 0 && w1 > 0 && w2 > 0) {
                double var = w0 * pow(m0 / w0 - gMean, 2) +
                             w1 * pow(m1 / w1 - gMean, 2) +
                             w2 * pow(m2 / w2 - gMean, 2);

                if (var > maxVar) {
                    maxVar = var;
                    bestTH1 = i;
                    bestTH2 = j;
                }
            }
        }
    }

    Mat out = Mat::zeros(img.size(), CV_8U);
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++) {
            uchar pixel = img.at<uchar>(x, y);
            if (pixel >= bestTH2)
                out.at<uchar>(x, y) = 255;
            else if (pixel >= bestTH1)
                out.at<uchar>(x, y) = 127;
        }

    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/fiore.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    if (src.empty()) return -1;

    Mat dst = otsu2k(src);

    imshow("Otsu2k", dst);
    waitKey(0);

    return 0;
}
