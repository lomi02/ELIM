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

    double totalPixels = img.rows * img.cols;
    for (double &val : hist)
        val /= totalPixels;

    double globalMean = 0.0;
    for (int i = 0; i < 256; i++)
        globalMean += i * hist[i];

    double prob = 0.0;
    double cumMean = 0.0;
    double maxVar = 0.0;
    int bestThreshold = 0;

    for (int k = 0; k < 256; k++) {
        prob += hist[k];
        cumMean += k * hist[k];
        double betweenVar = pow(globalMean * prob - cumMean, 2) / (prob * (1.0 - prob));
        if (betweenVar > maxVar) {
            maxVar = betweenVar;
            bestThreshold = k;
        }
    }

    Mat out;
    threshold(img, out, bestThreshold, 255, THRESH_BINARY);

    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/fiore.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    Mat dst = otsu(src);

    imshow("Otsu2k", dst);
    waitKey(0);

    return 0;
}
