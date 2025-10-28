#include <opencv2/opencv.hpp>
#include <stack>
using namespace cv;
using namespace std;

Mat regionGrowing(Mat &input) {
    Mat src = input.clone();

    int similarityThreshold = 5;
    double minAreaFactor = 0.01;
    uchar maxLabels = 100;

    int minArea = int(minAreaFactor * src.rows * src.cols);
    Mat labels = Mat::zeros(src.rows, src.cols, CV_8U);
    Mat regionMask = Mat::zeros(src.rows, src.cols, CV_8U);
    uchar currentLabel = 1;

    const Point neighbors[8] = {
        Point(1, 0), Point(1, -1), Point(0, -1), Point(-1, -1),
        Point(-1, 0), Point(-1, 1), Point(0, 1), Point(1, 1)
    };

    for (int x = 0; x < src.cols; x++) {
        for (int y = 0; y < src.rows; y++) {
            Point seed(x, y);
            if (labels.at<uchar>(seed) != 0) continue;

            stack<Point> points;
            points.push(seed);
            regionMask.setTo(0);

            while (!points.empty()) {
                Point p = points.top();
                points.pop();
                regionMask.at<uchar>(p) = 1;
                uchar centerVal = src.at<uchar>(p);

                for (int i = 0; i < 8; i++) {
                    Point q = p + neighbors[i];
                    if (q.x < 0 || q.x >= src.cols || q.y < 0 || q.y >= src.rows)
                        continue;
                    if (labels.at<uchar>(q) || regionMask.at<uchar>(q))
                        continue;
                    uchar neighVal = src.at<uchar>(q);
                    if (abs(int(centerVal) - int(neighVal)) < similarityThreshold) {
                        regionMask.at<uchar>(q) = 1;
                        points.push(q);
                    }
                }
            }

            int regionArea = int(sum(regionMask)[0]);
            if (regionArea > minArea) {
                labels += regionMask * currentLabel;
                if (++currentLabel > maxLabels) return labels;
            } else {
                labels += regionMask * 255;
            }
        }
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
