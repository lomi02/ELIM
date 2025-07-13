#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat region_growing(Mat &input, int similTH, Point seed) {
    Mat visited = Mat::zeros(input.size(), CV_8U);

    queue<Point> pixelQueue;
    pixelQueue.push(seed);
    visited.at<uchar>(seed) = 1;

    Point neighbors[8] = {
        Point(-1, -1),  Point(-1, 0),   Point(-1, 1),
        Point(0, -1),                   Point(0, 1),
        Point(1, -1),   Point(1, 0),    Point(1, 1)
    };

    Mat out = Mat::zeros(input.size(), CV_8U);
    while (!pixelQueue.empty()) {
        Point currentPx = pixelQueue.front();
        pixelQueue.pop();
        out.at<uchar>(currentPx) = 255;

        for (Point &offset: neighbors) {
            Point neighPx = currentPx + offset;

            if (neighPx.x >= 0 && neighPx.x < input.cols &&
                neighPx.y >= 0 && neighPx.y < input.rows &&
                visited.at<uchar>(neighPx) == 0 &&
                abs(input.at<uchar>(currentPx) - input.at<uchar>(neighPx)) <= similTH) {
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
