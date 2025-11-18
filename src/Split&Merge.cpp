#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

double smTH = 10;
int tSize = 8;
int mTH = 5;

class TNode {
public:
    Rect region;
    TNode *regions[4] = {nullptr};
    vector<TNode *> merged;
    bool isMerged[4] = {false};
    double stddev, mean;

    TNode(Rect R) : region(R) {}
};

TNode *split(Mat &img, Rect R) {
    TNode *root = new TNode(R);

    Scalar stddev, mean;
    meanStdDev(img(R), mean, stddev);
    root->mean = mean[0];
    root->stddev = stddev[0];

    if (R.width > tSize && root->stddev > smTH) {
        int h = R.height / 2, w = R.width / 2;
        root->regions[0] = split(img, Rect(R.x, R.y, h, w));
        root->regions[1] = split(img, Rect(R.x, R.y + w, h, w));
        root->regions[2] = split(img, Rect(R.x + h, R.y, h, w));
        root->regions[3] = split(img, Rect(R.x + h, R.y + w, h, w));
    }

    rectangle(img, R, Scalar(0));
    return root;
}

void merge(TNode *root) {
    if (root->region.width > tSize && root->stddev > smTH) {
        int mean[4];
        for (int i = 0; i < 4; i++)
            mean[i] = (int) root->regions[i]->mean;

        for (int i = 0; i < 4; i++) {
            int next = (i + 1) % 4;
            if (abs(mean[i] - mean[next]) < mTH) {
                root->merged.push_back(root->regions[i]);
                root->merged.push_back(root->regions[next]);
                root->isMerged[i] = root->isMerged[next] = true;

                int next2 = (i + 2) % 4, prev = (i + 3) % 4;
                if (abs(mean[next] - mean[next2]) < mTH) {
                    root->merged.push_back(root->regions[next2]);
                    root->isMerged[next2] = true;
                } else if (abs(mean[prev] - mean[i]) < mTH) {
                    root->merged.push_back(root->regions[prev]);
                    root->isMerged[prev] = true;
                }
            }
        }
        for (int i = 0; i < 4; i++)
            if (!root->isMerged[i])
                merge(root->regions[i]);
    } else
        root->merged.push_back(root);
}

void segment(TNode *root, Mat &img) {
    float val = 0;
    for (auto node: root->merged)
        val += node->mean;
    val /= root->merged.size();

    for (auto node: root->merged)
        img(node->region) = (int) val;

    for (int i = 0; i < 4; i++)
        if (!root->isMerged[i] && root->regions[i])
            segment(root->regions[i], img);
}

void SplitMerge(Mat &input) {
    Mat img = input.clone();
    GaussianBlur(img, img, Size(3, 3), 1, 1);

    int exponent = log(min(img.cols, img.rows)) / log(2);
    int quadSize = pow(2.0, (double) exponent);

    Rect square = Rect(0, 0, quadSize, quadSize);
    img = img(square).clone();

    Mat imgSeg = img.clone();

    TNode *root = split(img, Rect(0, 0, img.rows, img.cols));
    merge(root);
    segment(root, imgSeg);

    imshow("Quad Tree", img);
    imshow("Segmented", imgSeg);
    waitKey(0);
}

int main() {
    Mat src = imread("../immagini/foglia.png", IMREAD_GRAYSCALE);

    SplitMerge(src);

    return 0;
}
