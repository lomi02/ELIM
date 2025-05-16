#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define up_left 0    //  -----
#define up_right 1   //  |0|1|
#define low_left 3   //  |3|2|
#define low_right 2  //  -----

class qtNode {
    Rect region;
    qtNode *upper_left = nullptr;
    qtNode *upper_right = nullptr;
    qtNode *lower_left = nullptr;
    qtNode *lower_right = nullptr;
    vector<qtNode *> merged;
    vector<bool> isMerged = vector(4, false);
    double mean = -1;
    double stdDev = -1;

public:
    const Rect &getRegion() const { return region; }
    void setRegion(const Rect &region) { qtNode::region = region; }
    qtNode *getUpperLeft() const { return upper_left; }
    void setUpperLeft(qtNode *upperLeft) { upper_left = upperLeft; }
    qtNode *getUpperRight() const { return upper_right; }
    void setUpperRight(qtNode *upperRight) { upper_right = upperRight; }
    qtNode *getLowerLeft() const { return lower_left; }
    void setLowerLeft(qtNode *lowerLeft) { lower_left = lowerLeft; }
    qtNode *getLowerRight() const { return lower_right; }
    void setLowerRight(qtNode *lowerRight) { lower_right = lowerRight; }
    double getMean() const { return mean; }
    void setMean(double mean) { qtNode::mean = mean; }
    double getStdDev() const { return stdDev; }
    void setStdDev(double stdDev) { qtNode::stdDev = stdDev; }
    void pushOnMerged(qtNode *toPush) { merged.push_back(toPush); }
    vector<qtNode *> getMerged() { return merged; }
    void setIsMerged(int quadrant, bool flag) { isMerged.at(quadrant) = flag; }
    void setIsMergedAllFalse() { for (auto &&i: isMerged) i = false; }
    vector<bool> getIsMerged() { return isMerged; }
    explicit qtNode(Rect &region) { qtNode::region = region; }
};

bool satisfyPredicate(qtNode *node, double samTH) {
    return node->getStdDev() <= samTH;
}

bool regionIsDivisible(qtNode *node, int minRegSize) {
    return node->getRegion().width > minRegSize;
}

bool regionIsDivisible(Rect region, int minRegSize) {
    return region.width > minRegSize and region.height > minRegSize;
}

bool shouldBeMerged(qtNode *node1, qtNode *node2, double samTH) {
    return satisfyPredicate(node1, samTH) and satisfyPredicate(node2, samTH);
}

qtNode *split(Mat &img, Rect region, double samTH, int minRegSize = 2) {
    auto node = new qtNode(region);
    Mat subImg = img(region);
    Scalar mean, stdDev;
    meanStdDev(subImg, mean, stdDev);
    node->setMean(mean[0]);
    node->setStdDev(stdDev[0]);

    if (regionIsDivisible(region, minRegSize) and not satisfyPredicate(node, samTH)) {
        int halfWidth = region.width / 2;
        int halfHeight = region.height / 2;

        Rect upper_left(region.x, region.y, halfWidth, halfHeight);
        node->setUpperLeft(split(img, upper_left, samTH, minRegSize));

        Rect upper_right(region.x + halfWidth, region.y, halfWidth, halfHeight);
        node->setUpperRight(split(img, upper_right, samTH, minRegSize));

        Rect lower_left(region.x, region.y + halfHeight, halfWidth, halfHeight);
        node->setLowerLeft(split(img, lower_left, samTH, minRegSize));

        Rect lower_right(region.x + halfWidth, region.y + halfHeight, halfWidth, halfHeight);
        node->setLowerRight(split(img, lower_right, samTH, minRegSize));
    }

    return node;
}

void merge(qtNode *node, double samTH, int minRegSize) {
    if (not regionIsDivisible(node, minRegSize) or satisfyPredicate(node, samTH)) {
        node->pushOnMerged(node);
        node->setIsMergedAllFalse();
        return;
    }

    auto upper_left = node->getUpperLeft();
    auto upper_right = node->getUpperRight();
    auto lower_left = node->getLowerLeft();
    auto lower_right = node->getLowerRight();

    if (shouldBeMerged(upper_left, upper_right, samTH)) {
        node->pushOnMerged(upper_left);
        node->pushOnMerged(upper_right);
        node->setIsMerged(up_left, true);
        node->setIsMerged(up_right, true);

        if (shouldBeMerged(lower_left, lower_right, samTH)) {
            node->pushOnMerged(lower_right);
            node->pushOnMerged(lower_left);
            node->setIsMerged(low_right, true);
            node->setIsMerged(low_left, true);
        } else {
            merge(lower_left, samTH, minRegSize);
            merge(lower_right, samTH, minRegSize);
        }
    } else if (shouldBeMerged(upper_right, lower_right, samTH)) {
        node->pushOnMerged(upper_right);
        node->pushOnMerged(lower_right);
        node->setIsMerged(up_right, true);
        node->setIsMerged(low_right, true);

        if (shouldBeMerged(upper_left, lower_left, samTH)) {
            node->pushOnMerged(upper_left);
            node->pushOnMerged(lower_left);
            node->setIsMerged(up_left, true);
            node->setIsMerged(low_left, true);
        } else {
            merge(upper_left, samTH, minRegSize);
            merge(lower_left, samTH, minRegSize);
        }
    } else if (shouldBeMerged(upper_right, lower_right, samTH)) {
        node->pushOnMerged(lower_left);
        node->pushOnMerged(lower_right);
        node->setIsMerged(low_left, true);
        node->setIsMerged(low_right, true);

        if (shouldBeMerged(upper_left, upper_right, samTH)) {
            node->pushOnMerged(upper_left);
            node->pushOnMerged(upper_right);
            node->setIsMerged(up_left, true);
            node->setIsMerged(up_right, true);
        } else {
            merge(upper_left, samTH, minRegSize);
            merge(upper_right, samTH, minRegSize);
        }
    } else if (shouldBeMerged(upper_left, lower_left, samTH)) {
        node->pushOnMerged(upper_left);
        node->pushOnMerged(lower_left);
        node->setIsMerged(up_left, true);
        node->setIsMerged(low_left, true);

        if (shouldBeMerged(upper_right, lower_right, samTH)) {
            node->pushOnMerged(upper_right);
            node->pushOnMerged(lower_right);
            node->setIsMerged(up_right, true);
            node->setIsMerged(low_right, true);
        } else {
            merge(upper_right, samTH, minRegSize);
            merge(lower_right, samTH, minRegSize);
        }
    } else {
        merge(upper_left, samTH, minRegSize);
        merge(upper_right, samTH, minRegSize);
        merge(lower_right, samTH, minRegSize);
        merge(lower_left, samTH, minRegSize);
    }
}

void draw(Mat &img, qtNode *node) {
    if (node == nullptr)
        return;

    auto mergedVector = node->getMerged();

    if (mergedVector.empty()) {
        auto upper_left = node->getUpperLeft();
        auto upper_right = node->getUpperRight();
        auto lower_left = node->getLowerLeft();
        auto lower_right = node->getLowerRight();

        draw(img, upper_left);
        draw(img, upper_right);
        draw(img, lower_left);
        draw(img, lower_right);

        return;
    }

    double regionValue = 0.0;
    for (auto mergedNode: mergedVector)
        regionValue += mergedNode->getMean();
    regionValue /= (int) mergedVector.size();

    for (auto mergedNode: mergedVector)
        img(mergedNode->getRegion()) = (int) regionValue;

    if (mergedVector.size() <= 1)
        return;

    if (not node->getIsMerged().at(up_left)) {
        auto upper_left = node->getUpperLeft();
        draw(img, upper_left);
    }
    if (not node->getIsMerged().at(up_right)) {
        auto upper_right = node->getUpperRight();
        draw(img, upper_right);
    }
    if (not node->getIsMerged().at(low_right)) {
        auto lower_right = node->getLowerRight();
        draw(img, lower_right);
    }
    if (not node->getIsMerged().at(low_left)) {
        auto lower_left = node->getLowerLeft();
        draw(img, lower_left);
    }
}

Mat split_and_merge(Mat &inputImg, double samTH, int minRegSize = 2) {
    Mat img = inputImg.clone();

    int squareSize = max(img.rows, img.cols);
    double xScaling = (double) squareSize / img.cols;
    double yScaling = (double) squareSize / img.rows;

    Mat resized;
    resize(img, resized, Size(), xScaling, yScaling);

    auto startingRegion = Rect(0, 0, resized.cols, resized.rows);

    auto quadTreeRoot = split(resized, startingRegion, samTH, minRegSize);
    merge(quadTreeRoot, samTH, minRegSize);
    draw(resized, quadTreeRoot);

    double xScalingInverted = (double) img.cols / squareSize;
    double yScalingInverted = (double) img.rows / squareSize;

    Mat out;
    resize(resized, out, Size(), xScalingInverted, yScalingInverted);

    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/foglia.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    if (src.empty()) return -1;

    // Pre-processing: sfocatura gaussiana
    int blurSize = 3;
    double blurSigma = 1.0;
    GaussianBlur(src, src, Size(blurSize, blurSize), blurSigma, blurSigma);

    // Parametri per split and merge
    double samTH = 20.0; // Soglia di deviazione standard
    int minRegSize = 4; // Dimensione minima regione

    Mat dst = split_and_merge(src, samTH, minRegSize);

    return 0;
}
