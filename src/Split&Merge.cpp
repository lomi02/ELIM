#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

double smTH = 10;
int tSize = 8;
int mTH = 5;

class TNode {
public:
    Rect region;
    vector<TNode *> regions = vector<TNode *>(4, nullptr);
    vector<TNode *> merged;
    vector<bool> isMerged = vector<bool>(4, false);
    double stddev, mean, meanMerged;
    TNode(Rect R) { region = R; }
};

TNode *split(Mat &img, Rect R) {
    TNode *root = new TNode(R);

    Scalar stddev, mean;
    meanStdDev(img(R), mean, stddev);

    root->mean = mean[0];
    root->stddev = stddev[0];

    if (R.width > tSize && root->stddev > smTH) {
        root->regions[0] = split(img, Rect(R.x, R.y, R.height / 2, R.width / 2));
        root->regions[1] = split(img, Rect(R.x, R.y + R.width / 2, R.height / 2, R.width / 2));
        root->regions[2] = split(img, Rect(R.x + R.height / 2, R.y, R.height / 2, R.width / 2));
        root->regions[3] = split(img, Rect(R.x + R.height / 2, R.y + R.width / 2, R.height / 2, R.width / 2));
    }

    rectangle(img, R, Scalar(0));
    return root;
}

void merge(TNode *root) {
    if (root->region.width > tSize && root->stddev > smTH) {
        for (int i = 0; i < 4; i++) {
            if (abs((int) root->regions[i]->mean - (int) root->regions[(i + 1) % 4]->mean) < mTH) {
                root->merged.push_back(root->regions[i]);
                root->isMerged[i] = true;
                root->merged.push_back(root->regions[(i + 1) % 4]);
                root->isMerged[(i + 1) % 4] = true;

                if (abs((int) root->regions[(i + 1) % 4]->mean - (int) root->regions[(i + 2) % 4]->mean) < mTH) {
                    root->merged.push_back(root->regions[(i + 2) % 4]);
                    root->isMerged[(i + 2) % 4] = true;
                    break;
                }
                if (abs((int) root->regions[(i + 3) % 4]->mean - (int) root->regions[i]->mean) < mTH) {
                    root->merged.push_back(root->regions[(i + 3) % 4]);
                    root->isMerged[(i + 3) % 4] = true;
                    break;
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

    for (int i = 0; i < 4; i++) {
        if (!root->isMerged[i] && root->regions[i])
            segment(root->regions[i], img);
    }
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
