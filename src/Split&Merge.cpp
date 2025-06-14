#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

struct QNode {
    Rect rect;
    QNode *child[4] = {nullptr};
    double mean, dev;
    bool isLeaf = true;

    QNode(Rect r) : rect(r) {}
};

QNode *split(Mat &img, Rect rect, double splitTH, int minSize) {
    auto node = new QNode(rect);

    Scalar mean, dev;
    meanStdDev(img(rect), mean, dev);
    node->mean = mean[0];
    node->dev = dev[0];

    if (rect.width > minSize && node->dev > splitTH) {
        node->isLeaf = false;
        int halfW = rect.width / 2, halfH = rect.height / 2;

        node->child[0] = split(img, Rect(rect.x, rect.y, halfW, halfH), splitTH, minSize);
        node->child[1] = split(img, Rect(rect.x + halfW, rect.y, halfW, halfH), splitTH, minSize);
        node->child[2] = split(img, Rect(rect.x, rect.y + halfH, halfW, halfH), splitTH, minSize);
        node->child[3] = split(img, Rect(rect.x + halfW, rect.y + halfH, halfW, halfH), splitTH, minSize);
    }

    rectangle(img, rect, Scalar(0));
    return node;
}

void merge(QNode *node, double mergeTH) {
    if (node->isLeaf) return;

    for (int i = 0; i < 4; i++)
        merge(node->child[i], mergeTH);

    bool canMerge = true;
    for (int i = 0; i < 4; i++) {
        if (!node->child[i]->isLeaf) {
            canMerge = false;
            break;
        }
        if (abs(node->child[i]->mean - node->child[0]->mean) > mergeTH) {
            canMerge = false;
            break;
        }
    }

    if (canMerge) {
        node->isLeaf = true;
        node->mean = 0;
        for (int i = 0; i < 4; i++)
            node->mean += node->child[i]->mean;
        node->mean /= 4;
    }
}

void segment(QNode *node, Mat &img) {
    if (node->isLeaf)
        img(node->rect) = (int) node->mean;
    else
        for (int i = 0; i < 4; i++)
            segment(node->child[i], img);
}

void splitAndMerge(Mat img, double splitTH, int minSize, double mergeTH) {
    GaussianBlur(img, img, Size(5, 5), 0);

    int exponent = log(min(img.cols, img.rows)) / log(2);
    int size = pow(2.0, (double) exponent);
    img = img(Rect(0, 0, size, size)).clone();

    Mat qtree = img.clone();
    Mat segmented = img.clone();

    QNode *root = split(qtree, Rect(0, 0, size, size), splitTH, minSize);

    merge(root, mergeTH);
    segment(root, segmented);

    imshow("Quad Tree", qtree);
    imshow("Segmented", segmented);
    waitKey(0);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/foglia.png";
    Mat img = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    if (img.empty()) return -1;

    splitAndMerge(img, 10, 8, 5);
    return 0;
}
