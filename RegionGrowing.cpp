#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

bool inRange(Mat &img, Point neigh) {
    return neigh.x >= 0 and neigh.x < img.cols and neigh.y >= 0 and neigh.y < img.rows;
}

bool isSimilar(Mat &img, Point seed, Point neigh, int similTH) {
    int seedIntensity = img.at<uchar>(seed);
    int currIntensity = img.at<uchar>(neigh);
    return abs(seedIntensity - currIntensity) < similTH;
}

Mat region_growing(Mat &input, int similTH, Point seed = Point(0, 0)) {
    Mat img = input.clone();

    Mat out = Mat::zeros(img.size(), CV_8U);
    queue<Point> pixelQueue;
    pixelQueue.push(seed);
    while (!pixelQueue.empty()) {
        Point currentPx = pixelQueue.front();
        pixelQueue.pop();

        if (out.at<uchar>(currentPx) == 0) {
            out.at<uchar>(currentPx) = 255;

            Rect roi(currentPx.x - 1, currentPx.y - 1, 3, 3);
            for (int roi_x = roi.x; roi_x < roi.x + roi.height; ++roi_x)
                for (int roi_y = roi.y; roi_y < roi.y + roi.width; ++roi_y) {
                    Point neighPx(roi_x, roi_y);
                    if (inRange(img, neighPx) and isSimilar(img, seed, neighPx, similTH))
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

    int seedx = 20;
    int seedy = 40;
    Point seed(seedx, seedy);
    int similTH = 50;

    Mat dst = region_growing(src, similTH, seed);

    imshow("RegionGrowing", dst);
    waitKey(0);
    return 0;
}
