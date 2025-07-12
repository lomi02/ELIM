#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

double smThreshold = 10;
int minRegSize = 8;
int mThreshold = 1;

using namespace std;
using namespace cv;

class TNode {
public:
    Rect region;
    vector<TNode *> regions = vector<TNode *>(4, nullptr);
    vector<TNode *> merged;
    vector<bool> isMerged = vector(4, false);
    double stddev, mean, meanMerged;
    TNode(Rect R) { region = R; };
};

TNode *split(Mat &src, Rect R) {
    auto node = new TNode(R);
    Scalar stddev, mean;
    meanStdDev(src(R), mean, stddev);
    node->stddev = stddev[0];
    node->mean = mean[0];
    if (R.width > minRegSize && node->stddev > smThreshold) {
        node->regions[0] = split(src, Rect(R.x, R.y, R.height / 2, R.width / 2));
        node->regions[1] = split(src, Rect(R.x, R.y + R.width / 2, R.height / 2, R.width / 2));
        node->regions[2] = split(src, Rect(R.x + R.height / 2, R.y, R.height / 2, R.width / 2));
        node->regions[3] = split(src, Rect(R.x + R.height / 2, R.y + R.width / 2, R.height / 2, R.width / 2));
    }
    rectangle(src, R, Scalar(0));
    return node;
}

void merge(TNode *node) {
    if (node->region.width > minRegSize && node->stddev > smThreshold) {
        for (int i = 0; i < 4; i++)
            if (abs((int) node->regions[i]->mean - (int) node->regions[(i + 1) % 4]->mean) < mThreshold) {
                node->merged.push_back(node->regions[i]);
                node->isMerged[i] = true;
                node->merged.push_back(node->regions[(i + 1) % 4]);
                node->isMerged[(i + 1) % 4] = true;
                if (abs((int) node->regions[(i + 1) % 4]->mean - (int) node->regions[(i + 2) % 4]->mean) < mThreshold) {
                    node->merged.push_back(node->regions[(i + 2) % 4]);
                    node->isMerged[(i + 2) % 4] = true;
                    break;
                }
                if (abs((int) node->regions[(i + 3) % 4]->mean - (int) node->regions[i]->mean) < mThreshold) {
                    node->merged.push_back(node->regions[(i + 3) % 4]);
                    node->isMerged[(i + 3) % 4] = true;
                    break;
                }
            }
        for (int i = 0; i < 4; i++)
            if (!node->isMerged[i])
                merge(node->regions[i]);
    }
    else
        node->merged.push_back(node);
}

void segment(TNode *src, Mat &dst) {
    float val = 0;
    for (auto node: src->merged)
        val += node->mean;
    val /= src->merged.size();
    for (auto node: src->merged)
        dst(node->region) = (int) val;
    for (int i = 0; i < 4; i++)
        if (!src->isMerged[i] && src->regions[i])
            segment(src->regions[i], dst);
}

void splitAndMerge(Mat &src, Mat &output) {
    int exponent = log(min(src.rows, src.cols)) / log(2);
    int size = pow(2.0, double(exponent));
    Mat img = src(Rect(0, 0, size, size)).clone();

    GaussianBlur(img, img, Size(3, 3), 0, 0);
    TNode *node = split(img, Rect(0, 0, img.rows, img.cols));
    merge(node);

    output = src(Rect(0, 0, size, size)).clone();
    segment(node, output);

    imshow("Working Image", img);
    imshow("Output Image", output);
    waitKey(0);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/foglia.png";
    Mat dst, src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    splitAndMerge(src, dst);
    return 0;
}
