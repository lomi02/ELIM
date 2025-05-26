#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat kmeans(Mat &input, int k, int maxIterations, double deltaTH) {
    Mat img = input.clone();

    vector<uchar> pixels;
    for (int i = 0; i < input.rows; i++)
        for (int j = 0; j < input.cols; j++)
            pixels.push_back(img.at<uchar>(i, j));

    vector<uchar> centroids(k);
    for (int i = 0; i < k; i++)
        centroids[i] = pixels[rand() % pixels.size()];

    vector<int> labels(pixels.size());
    bool centroidsChanged = true;
    int iterations = 0;

    while (centroidsChanged && iterations < maxIterations) {
        centroidsChanged = false;

        for (int i = 0; i < pixels.size(); i++) {
            int minDist = INT_MAX, nearestCluster = 0;

            for (int cluster = 0; cluster < k; cluster++) {
                int dist = abs(pixels[i] - centroids[cluster]);

                if (dist < minDist) {
                    minDist = dist;
                    nearestCluster = cluster;
                }
            }
            labels[i] = nearestCluster;
        }

        for (int cluster = 0; cluster < k; cluster++) {
            double sum = 0, count = 0;
            for (int i = 0; i < pixels.size(); i++) {
                if (labels[i] == cluster) {
                    sum += pixels[i];
                    count++;
                }
            }
            if (count > 0) {
                double newCenter = sum / count;
                if (abs(newCenter - centroids[cluster]) > deltaTH) {
                    centroids[cluster] = newCenter;
                    centroidsChanged = true;
                }
            }
        }
        iterations++;
    }

    Mat out(img.size(), CV_8U);
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            out.at<uchar>(i, j) = centroids[labels[i * img.cols + j]];

    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/splash.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    int k = 3;
    int maxIterations = 30;
    double deltaTH = 1;

    Mat dst = kmeans(src, k, maxIterations, deltaTH);

    imshow("K-means", dst);
    waitKey(0);

    return 0;
}
