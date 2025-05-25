#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

bool inRange(const Mat &img, Point neigh) {
    return neigh.x >= 0 && neigh.x < img.cols
           && neigh.y >= 0 && neigh.y < img.rows;
}

bool isSimilar(const Mat &img, Point p1, Point p2, int similTH) {
    return abs(img.at<uchar>(p1) - img.at<uchar>(p2)) <= similTH;
}

Mat region_growing(const Mat &input, int similTH, Point seed) {
    Mat img = input.clone();
    Mat out = Mat::zeros(img.size(), CV_8U);

    Mat visited = Mat::zeros(input.size(), CV_8U);
    queue<Point> pixelQueue;
    pixelQueue.push(seed);
    visited.at<uchar>(seed) = 1;

    Point neighbors[] = {
        Point(0, -1), Point(-1, 0),
        Point(1, 0),  Point(0, 1)
    };

    while (!pixelQueue.empty()) {
        Point currentPx = pixelQueue.front();
        pixelQueue.pop();
        out.at<uchar>(currentPx) = 255;

        for (const Point &offset: neighbors) {
            Point neighPx = currentPx + offset;
            if (inRange(input, neighPx) &&
                visited.at<uchar>(neighPx) == 0 &&
                isSimilar(input, currentPx, neighPx, similTH)) {
                visited.at<uchar>(neighPx) = 1;
                pixelQueue.push(neighPx);
                }
        }
    }

    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/splash.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    Point seed(src.cols / 2, src.rows / 2);
    int similTH = 10;

    Mat dst = region_growing(src, similTH, seed);

    imshow("RegionGrowing", dst);
    waitKey(0);
    return 0;
}
