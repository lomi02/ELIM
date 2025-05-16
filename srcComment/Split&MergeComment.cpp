#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define up_left 0    //  -----
#define up_right 1   //  |0|1|
#define low_left 3   //  |3|2|
#define low_right 2  //  -----

/**
 * @class qtNode
 * @brief Rappresenta un nodo in una struttura dati Quadtree.
 */
class qtNode {
    Rect region;    // Regione rettangolare rappresentata dal nodo

    // Puntatori ai 4 figli (quadranti)
    qtNode * upper_left = nullptr;
    qtNode * upper_right = nullptr;
    qtNode * lower_left = nullptr;
    qtNode * lower_right = nullptr;

    vector<qtNode *> merged;    // Nodi uniti in merge
    vector<bool> isMerged = vector(4, false);   // Flag merge per quadrante

    double mean = -1;       // Valore medio della regione
    double stdDev = -1;     // Deviazione standard della regione

public:

    // Metodi getter e setter
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
    void setIsMergedAllFalse() { for (auto && i : isMerged) i = false; }
    vector<bool> getIsMerged() { return isMerged; }

    explicit qtNode(Rect &region) { qtNode::region = region; }
};

/**
 * Verifica se un nodo soddisfa il predicato (deviazione standard <= soglia).
 *
 * @param node Il nodo da verificare
 * @param samTH La soglia per il predicato
 * @return True se il nodo soddisfa il predicato, altrimenti False
 */
bool satisfyPredicate(qtNode *node, double samTH) {
    return node->getStdDev() <= samTH;
}

/**
 * Verifica se una regione è divisibile in sotto-regioni.
 *
 * @param node Il nodo rappresentante la regione
 * @param minRegSize La dimensione minima delle regioni per la divisibilità
 * @return True se la regione è divisibile, altrimenti False
 */
bool regionIsDivisible(qtNode *node, int minRegSize) {
    return node->getRegion().width > minRegSize;
}

/**
 * Verifica se una regione è divisibile in sotto-regioni.
 *
 * @param region La regione rappresentata come cv::Rect
 * @param minRegSize La dimensione minima delle regioni per la divisibilità
 * @return True se la regione è divisibile, altrimenti False
 */
bool regionIsDivisible(Rect region, int minRegSize) {
    return region.width > minRegSize and region.height > minRegSize;
}

/**
 * Verifica se due regioni dovrebbero essere unite in una regione più grande.
 *
 * @param node1 Il nodo rappresentante la prima regione
 * @param node2 Il nodo rappresentante la seconda regione
 * @param samTH La soglia per il predicato
 * @return True se le regioni dovrebbero essere unite, altrimenti False
 */
bool shouldBeMerged(qtNode *node1, qtNode *node2, double samTH) {
    return satisfyPredicate(node1, samTH) and satisfyPredicate(node2, samTH);
}

/**
 * Esegue la divisione ricorsiva (split) di una regione immagine usando un Quadtree.
 *
 * @param img L'immagine in input
 * @param region La regione da dividere
 * @param samTH La soglia per lo splitting
 * @param minRegSize La dimensione minima delle regioni
 * @return Il nodo radice del Quadtree
 */
qtNode * split(Mat &img, Rect region, double samTH, int minRegSize = 2) {

    // Crea un nuovo nodo del Quadtree per questa regione
    auto node = new qtNode(region);

    // Calcola media e deviazione standard della regione
    Mat subImg = img(region);
    Scalar mean, stdDev;
    meanStdDev(subImg, mean, stdDev);

    node->setMean(mean[0]);
    node->setStdDev(stdDev[0]);

    // Se la regione è divisibile e non soddisfa il predicato
    if (regionIsDivisible(region, minRegSize) and not satisfyPredicate(node, samTH)) {

        // Calcola le dimensioni delle sotto-regioni
        int halfWidth = region.width / 2;
        int halfHeight = region.height / 2;

        // Divide la regione in 4 quadranti e procede ricorsivamente
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

/**
 * Esegue l'unione (merge) delle regioni nel Quadtree in base alla soglia.
 *
 * @param node Il nodo corrente da unire
 * @param samTH La soglia per l'unione
 * @param minRegSize La dimensione minima delle regioni
 */
void merge(qtNode *node, double samTH, int minRegSize) {

    // Se la regione non è divisibile o soddisfa il predicato
    if (not regionIsDivisible(node, minRegSize) or satisfyPredicate(node, samTH)) {
        node->pushOnMerged(node);
        node->setIsMergedAllFalse();
        return;
    }

    // Ottieni i puntatori ai 4 nodi figli
    auto upper_left = node->getUpperLeft();
    auto upper_right = node->getUpperRight();
    auto lower_left = node->getLowerLeft();
    auto lower_right = node->getLowerRight();

    // Verifica unione lungo la linea SUPERIORE
    if (shouldBeMerged(upper_left, upper_right, samTH)) {
        node->pushOnMerged(upper_left);
        node->pushOnMerged(upper_right);
        node->setIsMerged(up_left, true);
        node->setIsMerged(up_right, true);

        // Verifica se unire anche la linea INFERIORE
        if (shouldBeMerged(lower_left, lower_right, samTH)) {
            node->pushOnMerged(lower_right);
            node->pushOnMerged(lower_left);
            node->setIsMerged(low_right, true);
            node->setIsMerged(low_left, true);
        }
        else {
            merge(lower_left, samTH, minRegSize);
            merge(lower_right, samTH, minRegSize);
        }
    }
    // Verifica unione lungo la linea DESTRA
    else if (shouldBeMerged(upper_right, lower_right, samTH)) {
        node->pushOnMerged(upper_right);
        node->pushOnMerged(lower_right);
        node->setIsMerged(up_right, true);
        node->setIsMerged(low_right, true);

        // Verifica se unire anche la linea SINISTRA
        if (shouldBeMerged(upper_left, lower_left, samTH)) {
            node->pushOnMerged(upper_left);
            node->pushOnMerged(lower_left);
            node->setIsMerged(up_left, true);
            node->setIsMerged(low_left, true);
        }
        else {
            merge(upper_left, samTH, minRegSize);
            merge(lower_left, samTH, minRegSize);
        }
    }
    // Verifica unione lungo la linea INFERIORE
    else if (shouldBeMerged(upper_right, lower_right, samTH)) {
        node->pushOnMerged(lower_left);
        node->pushOnMerged(lower_right);
        node->setIsMerged(low_left, true);
        node->setIsMerged(low_right, true);

        // Verifica se unire anche la linea SUPERIORE
        if (shouldBeMerged(upper_left, upper_right, samTH)) {
            node->pushOnMerged(upper_left);
            node->pushOnMerged(upper_right);
            node->setIsMerged(up_left, true);
            node->setIsMerged(up_right, true);
        }
        else {
            merge(upper_left, samTH, minRegSize);
            merge(upper_right, samTH, minRegSize);
        }
    }
    // Verifica unione lungo la linea SINISTRA
    else if (shouldBeMerged(upper_left, lower_left, samTH)) {
        node->pushOnMerged(upper_left);
        node->pushOnMerged(lower_left);
        node->setIsMerged(up_left, true);
        node->setIsMerged(low_left, true);

        // Verifica se unire anche la linea DESTRA
        if (shouldBeMerged(upper_right, lower_right, samTH)) {
            node->pushOnMerged(upper_right);
            node->pushOnMerged(lower_right);
            node->setIsMerged(up_right, true);
            node->setIsMerged(low_right, true);
        }
        else {
            merge(upper_right, samTH, minRegSize);
            merge(lower_right, samTH, minRegSize);
        }
    }
    // Se nessuna condizione è soddisfatta, applica merge ricorsivamente su tutti i figli
    else {
        merge(upper_left, samTH, minRegSize);
        merge(upper_right, samTH, minRegSize);
        merge(lower_right, samTH, minRegSize);
        merge(lower_left, samTH, minRegSize);
    }
}

/**
 * Disegna le regioni unite sull'immagine.
 *
 * @param img L'immagine su cui disegnare
 * @param node Il nodo del Quadtree
 */
void draw(Mat &img, qtNode * node) {
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

    // Calcola il valore medio delle regioni unite
    double regionValue = 0.0;
    for (auto mergedNode : mergedVector)
        regionValue += mergedNode->getMean();
    regionValue /= (int) mergedVector.size();

    // Assegna il valore medio a tutte le regioni unite
    for (auto mergedNode : mergedVector)
        img(mergedNode->getRegion()) = (int) regionValue;

    if (mergedVector.size() <= 1)
        return;

    // Disegna ricorsivamente i quadranti non uniti
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

/**
 * Esegue la segmentazione Split and Merge sull'immagine.
 *
 * @param inputImg L'immagine in input
 * @param samTH La soglia per split e merge
 * @param minRegSize La dimensione minima delle regioni
 * @return L'immagine segmentata
 */
Mat split_and_merge(Mat & inputImg, double samTH, int minRegSize = 2) {
    Mat img = inputImg.clone();

    // Ridimensiona l'immagine a forma quadrata per l'algoritmo
    int squareSize = max(img.rows, img.cols);
    double xScaling = (double) squareSize / img.cols;
    double yScaling = (double) squareSize / img.rows;

    Mat resized;
    resize(img, resized, Size(), xScaling, yScaling);

    // Applica split and merge
    auto startingRegion = Rect(0, 0, resized.cols, resized.rows);

    auto quadTreeRoot = split(resized, startingRegion, samTH, minRegSize);
    merge(quadTreeRoot, samTH, minRegSize);
    draw(resized, quadTreeRoot);

    // Riporta l'immagine alla dimensione originale
    double xScalingInverted = (double) img.cols / squareSize;
    double yScalingInverted = (double) img.rows / squareSize;

    Mat out;
    resize(resized, out, Size(), xScalingInverted, yScalingInverted);

    return out;
}
