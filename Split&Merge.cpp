#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

const float MIN_REGION_SIZE = 8;
const int STDDEV_THRESHOLD = 6;

// 1. Definizione della struttura dati per la suddivisione dell'immagine
class TNode {
public:
    Rect region;
    TNode *UL, *UR, *LL, *LR;   //UpperLeft - UpperRight - LowerLeft - LowerRight
    vector<TNode *> mergedRegions;
    vector<bool> mergedFlags;
    double stddev, mean;    // stddev = Standard Deviation

    TNode(Rect region) : region(region), UL(nullptr), UR(nullptr), LL(nullptr), LR(nullptr), mergedFlags(4, false) {}

    void addMergedRegion(TNode *region) { mergedRegions.push_back(region); }
    void setMergedFlag(int index) { mergedFlags[index] = true; }
};

// 2. Funzione per dividere ricorsivamente l'immagine in regioni più piccole
TNode *split(Mat &src, Rect region) {
    auto root = new TNode(region);

    // 2.1 Calcola la media e la deviazione standard dell'area corrente
    Scalar mean, stddev;
    meanStdDev(src(region), mean, stddev);
    root->mean = mean[0];
    root->stddev = stddev[0];

    // 2.2 Se la regione è abbastanza grande e ha un'alta deviazione standard, la divide ulteriormente
    if (region.width > MIN_REGION_SIZE && root->stddev > STDDEV_THRESHOLD) {
        int halfWidth = region.width / 2, halfHeight = region.height / 2;
        root->UL = split(src, Rect(region.x, region.y, halfWidth, halfHeight));
        root->UR = split(src, Rect(region.x + halfWidth, region.y, halfWidth, halfHeight));
        root->LL = split(src, Rect(region.x, region.y + halfHeight, halfWidth, halfHeight));
        root->LR = split(src, Rect(region.x + halfWidth, region.y + halfHeight, halfWidth, halfHeight));
    }

    // 2.3 Disegna il rettangolo per visualizzare la suddivisione
    rectangle(src, region, Scalar(0));
    return root;
}

// 3. Funzione per unire regioni in base alla deviazione standard
void merge(TNode *root) {
    if (root->region.width > MIN_REGION_SIZE && root->stddev > STDDEV_THRESHOLD) {

        // 3.1 Funzione lambda per controllare e unire regioni adiacenti
        auto checkAndMerge = [&](TNode *a, TNode *b, int flagA, int flagB) {
            if (a->stddev <= STDDEV_THRESHOLD && b->stddev <= STDDEV_THRESHOLD) {
                root->addMergedRegion(a);
                root->setMergedFlag(flagA);
                root->addMergedRegion(b);
                root->setMergedFlag(flagB);
                return true;
            }
            return false;
        };

        // 3.2 Controlla e unisce le regioni adiacenti se possibile
        if (!checkAndMerge(root->UL, root->UR, 0, 1)) merge(root->UL), merge(root->UR);
        if (!checkAndMerge(root->UR, root->LR, 1, 2)) merge(root->UR), merge(root->LR);
        if (!checkAndMerge(root->LL, root->LR, 3, 2)) merge(root->LL), merge(root->LR);
        if (!checkAndMerge(root->UL, root->LL, 0, 3)) merge(root->UL), merge(root->LL);
    } else {

        // 3.3 Se non si possono unire ulteriormente, si segna come un'unica regione
        root->addMergedRegion(root);
        for (int i = 0; i < 4; i++) root->setMergedFlag(i);
    }
}

// 4. Funzione per assegnare il valore medio alle regioni segmentate
void segment(TNode *root, Mat &src) {
    if (root->mergedRegions.empty()) {

        // 4.1 Se non ci sono regioni unite, segmenta i figli
        segment(root->UL, src);
        segment(root->UR, src);
        segment(root->LR, src);
        segment(root->LL, src);

    } else {

        // 4.2 Calcola il valore medio delle regioni unite
        double meanValue = 0;
        for (auto region: root->mergedRegions) meanValue += region->mean;
        meanValue /= root->mergedRegions.size();

        // 4.3 Applica il valore medio alle regioni unite
        for (auto region: root->mergedRegions) src(region->region) = static_cast<int>(meanValue);

        // 4.4 Controlla quali regioni sono state unite e segmenta le rimanenti
        if (root->mergedRegions.size() > 1) {
            if (!root->mergedFlags[0]) segment(root->UL, src);
            if (!root->mergedFlags[1]) segment(root->UR, src);
            if (!root->mergedFlags[2]) segment(root->LR, src);
            if (!root->mergedFlags[3]) segment(root->LL, src);
        }
    }
}

// 5. Funzione principale per applicare il metodo split-and-merge
void Split_Merge(Mat &src, Mat &segmentedDst) {

    // 5.1 Calcola la dimensione più grande possibile come potenza di 2
    int size = pow(2, static_cast<int>(log(min(src.rows, src.cols)) / log(2)));

    // 5.2 Ritaglia l'immagine
    Mat croppedSrc = src(Rect(0, 0, size, size)).clone();

    // 5.3 Esegue la suddivisione e fusione delle regioni
    TNode *root = split(croppedSrc, Rect(0, 0, size, size));
    merge(root);

    // 5.4 Segmenta l'immagine finale
    segmentedDst = croppedSrc.clone();
    segment(root, segmentedDst);

    // 5.5 Mostra le immagini risultanti
    imshow("CroppedSrc", croppedSrc);       // Immagine croppata e sezionata in base ai livelli di intensità (split)
    imshow("SegmentedDst", segmentedDst);   // Immagine nel quale ogni nodo ottiene il valore medio di intensità dell'area (segment)
    waitKey(0);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/foglia.png";
    Mat dst, src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    Split_Merge(src, dst);

    return 0;
}
