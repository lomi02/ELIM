#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat kmeans(Mat &input, int k) {
    Mat img = input.clone();
    srand(static_cast<unsigned>(time(nullptr)));

    vector<uchar> centroids(k);
    for (int i = 0; i < k; i++) {
        int x = rand() % img.rows;
        int y = rand() % img.cols;
        centroids[i] = img.at<uchar>(x, y);
    }

    vector<vector<Point> > clusters(k);

    for (int iter = 0; iter < 50; iter++) {
        for (size_t i = 0; i < clusters.size(); i++)
            clusters[i].clear();

        for (int x = 0; x < img.rows; x++)
            for (int y = 0; y < img.cols; y++) {
                uchar pixel = img.at<uchar>(x, y);

                int best = 0;
                for (int i = 1; i < k; i++)
                    if (abs(static_cast<int>(centroids[i]) - pixel) <
                        abs(static_cast<int>(centroids[best]) - pixel))
                        best = i;
                clusters[best].push_back(Point(x, y));
            }

        bool changed = false;
        for (int i = 0; i < k; i++) {
            if (clusters[i].empty())
                continue;

            int sum = 0;
            for (size_t j = 0; j < clusters[i].size(); j++)
                sum += img.at<uchar>(clusters[i][j].x, clusters[i][j].y);

            uchar newCentroid = static_cast<uchar>(sum / clusters[i].size());
            if (abs(static_cast<int>(newCentroid) - centroids[i]) > 0)
                changed = true;
            centroids[i] = newCentroid;
        }

        if (!changed)
            break;
    }

    Mat out = img.clone();
    for (int i = 0; i < k; i++)
        for (size_t j = 0; j < clusters[i].size(); j++)
            out.at<uchar>(clusters[i][j].x, clusters[i][j].y) = centroids[i];

    return out;
}

int main() {
    Mat src = imread("../immagini/splash.png", IMREAD_GRAYSCALE);

    int k = 3;

    Mat dst = kmeans(src, k);

    imshow("K-means", dst);
    waitKey(0);

    return 0;
}
