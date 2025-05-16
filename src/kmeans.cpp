#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat kmeans_gray(Mat &input, int numberOfClusters, int maxIterations, double deltaTH = 1.0) {
    Mat img = input.clone();

    srand(time(nullptr) + 1);
    vector<uchar> centres(numberOfClusters);
    for (int i = 0; i < numberOfClusters; ++i) {
        int x = rand() % img.cols;
        int y = rand() % img.rows;
        centres.at(i) = img.at<uchar>(Point(x, y));
    }

    int iterations = 0;
    int closestIndex = 0;
    bool isCentreUpdated = true;

    vector<vector<Point> > clusters(numberOfClusters);
    Mat clusteredImg = img.clone();

    while (isCentreUpdated || iterations < maxIterations) {
        isCentreUpdated = false;
        for (int i = 0; i < numberOfClusters; ++i)
            clusters.at(i).clear();

        for (int y = 0; y < img.rows; ++y)
            for (int x = 0; x < img.cols; ++x) {
                int minDistance = INFINITY;
                for (int i = 0; i < numberOfClusters; ++i) {
                    int currentDistance = abs(centres.at(i) - img.at<uchar>(Point(x, y)));
                    if (currentDistance < minDistance) {
                        minDistance = currentDistance;
                        closestIndex = i;
                    }
                }
                clusters.at(closestIndex).emplace_back(x, y);
                clusteredImg.at<uchar>(Point(x, y)) = centres.at(closestIndex);
            }

        for (int i = 0; i < numberOfClusters; ++i)
            if (!clusters.at(i).empty()) {
                int intensitySum = 0;
                for (auto &point: clusters.at(i))
                    intensitySum += img.at<uchar>(point);

                double currentMean = intensitySum / clusters.at(i).size();
                int delta = cvRound(std::abs(currentMean - centres.at(i)));

                if (delta > deltaTH) {
                    centres.at(i) = cvRound(currentMean);
                    isCentreUpdated = true;
                }
            }
        iterations++;
    }
    return clusteredImg;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/splash.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    int k = 3;
    int maxIterations = 30;
    double deltaTH = 1.0;

    Mat dst = kmeans_gray(src, k, maxIterations, deltaTH);

    imshow("K-means", dst);
    waitKey(0);

    return 0;
}
