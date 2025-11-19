#include <opencv2/opencv.hpp>
#include <stack>
using namespace cv;
using namespace std;

Mat regionGrowing(Mat &input) {
    Mat img = input.clone();

    int simTH = 5;
    double minAreaFactor = 0.01;
    uchar maxLabels = 100;

    int minArea = int(minAreaFactor * img.rows * img.cols);
    Mat labels = Mat::zeros(img.rows, img.cols, CV_8U);
    Mat regionMask = Mat::zeros(img.rows, img.cols, CV_8U);
    uchar currentLabel = 1;

    const Point neighbors[8] = {
        Point(1, 0), Point(1, -1), Point(0, -1), Point(-1, -1),
        Point(-1, 0), Point(-1, 1), Point(0, 1), Point(1, 1)
    };

    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++) {
            Point seed(y, x);

            if (labels.at<uchar>(seed) != 0)
                continue;

            stack<Point> points;
            points.push(seed);
            regionMask.setTo(0);

            while (!points.empty()) {
                Point current = points.top();
                points.pop();
                regionMask.at<uchar>(current) = 1;
                uchar currentVal = img.at<uchar>(current);

                for (int i = 0; i < 8; i++) {
                    Point neighbor = current + neighbors[i];

                    if (neighbor.x < 0 || neighbor.x >= img.cols || neighbor.y < 0 || neighbor.y >= img.rows)
                        continue;

                    if (labels.at<uchar>(neighbor) || regionMask.at<uchar>(neighbor))
                        continue;

                    uchar neighborVal = img.at<uchar>(neighbor);
                    if (abs(int(currentVal) - int(neighborVal)) < simTH) {
                        regionMask.at<uchar>(neighbor) = 1;
                        points.push(neighbor);
                    }
                }
            }

            int regionArea = int(sum(regionMask)[0]);
            if (regionArea > minArea) {
                labels += regionMask * currentLabel;
                if (currentLabel++ > maxLabels)
                    return labels;
            } else
                labels += regionMask * 255;
        }

    return labels;
}

int main() {
    Mat src = imread("../immagini/splash.png", IMREAD_GRAYSCALE);

    Mat dst = regionGrowing(src);

    imshow("RegionGrowing", dst);
    waitKey(0);

    return 0;
}
