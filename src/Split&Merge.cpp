#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

double smThreshold = 10;
int minRegSize = 8;
int mThreshold = 1;

class TNode {
public:
    Rect region;
    vector<TNode *> regions;
    vector<TNode *> merged;
    vector<bool> isMerged;
    double stddev, mean;

    TNode(Rect R) : region(R), regions(4, nullptr), isMerged(4, false) {
    }

    bool shouldSplit() const {
        return region.width > minRegSize && stddev > smThreshold;
    }
};

TNode *split(Mat &src, Rect R) {
    TNode *node = new TNode(R);
    Scalar stddev, mean;
    meanStdDev(src(R), mean, stddev);
    node->stddev = stddev[0];
    node->mean = mean[0];

    if (node->shouldSplit()) {
        int halfH = R.height / 2;
        int halfW = R.width / 2;
        node->regions[0] = split(src, Rect(R.x, R.y, halfH, halfW));
        node->regions[1] = split(src, Rect(R.x, R.y + halfW, halfH, halfW));
        node->regions[2] = split(src, Rect(R.x + halfH, R.y, halfH, halfW));
        node->regions[3] = split(src, Rect(R.x + halfH, R.y + halfW, halfH, halfW));
    }

    rectangle(src, R, Scalar(0));
    return node;
}

bool canMerge(TNode *a, TNode *b) {
    return abs((int) a->mean - (int) b->mean) < mThreshold;
}

void merge(TNode *node) {
    if (!node->shouldSplit()) {
        node->merged.push_back(node);
        return;
    }

    for (int i = 0; i < 4; i++) {
        if (canMerge(node->regions[i], node->regions[(i + 1) % 4])) {
            node->merged.push_back(node->regions[i]);
            node->isMerged[i] = true;
            node->merged.push_back(node->regions[(i + 1) % 4]);
            node->isMerged[(i + 1) % 4] = true;

            if (canMerge(node->regions[(i + 1) % 4], node->regions[(i + 2) % 4])) {
                node->merged.push_back(node->regions[(i + 2) % 4]);
                node->isMerged[(i + 2) % 4] = true;
                break;
            }
            if (canMerge(node->regions[(i + 3) % 4], node->regions[i])) {
                node->merged.push_back(node->regions[(i + 3) % 4]);
                node->isMerged[(i + 3) % 4] = true;
                break;
            }
        }
    }

    for (int i = 0; i < 4; i++)
        if (!node->isMerged[i])
            merge(node->regions[i]);
}

void segment(TNode *node, Mat &dst) {
    float val = 0;
    for (size_t i = 0; i < node->merged.size(); i++)
        val += node->merged[i]->mean;
    val /= node->merged.size();

    for (size_t i = 0; i < node->merged.size(); i++)
        dst(node->merged[i]->region) = (int) val;

    for (int i = 0; i < 4; i++)
        if (!node->isMerged[i] && node->regions[i])
            segment(node->regions[i], dst);
}

void splitAndMerge(Mat &src, Mat &output, Mat &working) {
    int exponent = log(min(src.rows, src.cols)) / log(2);
    int size = pow(2.0, (double) exponent);
    working = src(Rect(0, 0, size, size)).clone();

    GaussianBlur(working, working, Size(3, 3), 0, 0);
    TNode *node = split(working, Rect(0, 0, working.rows, working.cols));
    merge(node);

    output = src(Rect(0, 0, size, size)).clone();
    segment(node, output);
}

int main() {
    Mat src = imread("../immagini/foglia.png", IMREAD_GRAYSCALE);

    Mat dst, working;
    splitAndMerge(src, dst, working);

    imshow("Working Image", working);
    imshow("Output Image", dst);
    waitKey(0);

    return 0;
}
