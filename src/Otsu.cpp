#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

Mat otsu(Mat &input) {
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

    double w = 0.0;
    double cMean = 0.0;
    double maxVar = 0.0;
    int bestTH = 0;

    for (int k = 0; k < 256; k++) {
        w += hist[k];
        cMean += k * hist[k];
        double var = pow(gMean * w - cMean, 2) / (w * (1.0 - w));
        if (var > maxVar) {
            maxVar = var;
            bestTH = k;
        }
    }

    Mat out;
    threshold(img, out, bestTH, 255, THRESH_BINARY);

    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/fiore.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    if (src.empty()) return -1;

    Mat dst = otsu(src);

    imshow("Otsu", dst);
    waitKey(0);

    return 0;
}
