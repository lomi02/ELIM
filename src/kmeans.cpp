#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat kmeans(Mat &input, int k) {
    Mat img = input.clone();
    srand(time(nullptr));

    vector<uchar> centroids(k);
    for (int i = 0; i < k; i++) {
        int x = rand() % img.rows;
        int y = rand() % img.cols;
        centroids[i] = img.at<uchar>(x, y);
    }

    vector<vector<Point> > clusters(k);

    for (int iter = 0; iter < 50; iter++) {
        for (auto &cluster: clusters) cluster.clear();

        for (int x = 0; x < img.rows; x++)
            for (int y = 0; y < img.cols; y++) {
                uchar pixel = img.at<uchar>(x, y);

                int best = 0;
                for (int i = 1; i < k; i++)
                    if (abs(centroids[i] - pixel) < abs(centroids[best] - pixel))
                        best = i;
                clusters[best].push_back(Point(x, y));
            }

        bool changed = false;
        for (int i = 0; i < k; i++) {
            if (clusters[i].empty())
                continue;

            int sum = 0;
            for (Point &p: clusters[i])
                sum += img.at<uchar>(p.x, p.y);

            uchar newCentroid = sum / clusters[i].size();
            if (abs(newCentroid - centroids[i]) > 0.01)
                changed = true;
            centroids[i] = newCentroid;
        }

        if (!changed)
            break;
    }

    Mat out = img.clone();
    for (int i = 0; i < k; i++)
        for (const Point &p: clusters[i])
            out.at<uchar>(p.x, p.y) = centroids[i];

    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/splash.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    int k = 3;

    Mat dst = kmeans(src, k);

    imshow("K-means", dst);
    waitKey(0);

    return 0;
}
