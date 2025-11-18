#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
 * Parametri dell’algoritmo Split & Merge:
 *
 * smTH : soglia della deviazione standard per permettere la suddivisione
 * tSize: dimensione minima consentita per continuare a dividere una regione
 * mTH  : differenza massima tra medie che consente di fondere due regioni adiacenti
 */
double smTH = 10;
int tSize = 8;
int mTH = 5;

/**
 * Nodo del Quad-Tree.
 * Rappresenta una regione dell’immagine e include:
 * - Il rettangolo della regione
 * - I quattro eventuali sotto-nodi risultanti dalla suddivisione
 * - L’elenco delle regioni fuse
 * - Flag che indicano quali sotto-regioni sono state fuse
 * - Le statistiche della regione (media e deviazione standard)
 */
class TNode {
public:
    Rect region;                // Regione riferita al nodo
    TNode *regions[4] = {nullptr};  // Sotto-regioni
    vector<TNode *> merged;     // Regioni fuse
    bool isMerged[4] = {false}; // Indicatori di fusione
    double stddev, mean;        // Statistiche della regione

    TNode(Rect R) : region(R) {}
};

/**
 * Suddivide ricorsivamente la regione in 4 quadranti.
 * Per ogni nodo:
 * - calcola media e deviazione standard
 * - se la regione è sufficientemente grande e non omogenea → suddivide
 * - disegna i confini per la visualizzazione del Quad-Tree
 */
TNode *split(Mat &img, Rect R) {
    TNode *root = new TNode(R);

    Scalar stddev, mean;
    meanStdDev(img(R), mean, stddev);
    root->mean = mean[0];
    root->stddev = stddev[0];

    // Condizione di split
    if (R.width > tSize && root->stddev > smTH) {
        int h = R.height / 2, w = R.width / 2;

        root->regions[0] = split(img, Rect(R.x,         R.y,         h, w));
        root->regions[1] = split(img, Rect(R.x,         R.y + w,     h, w));
        root->regions[2] = split(img, Rect(R.x + h,     R.y,         h, w));
        root->regions[3] = split(img, Rect(R.x + h,     R.y + w,     h, w));
    }

    rectangle(img, R, Scalar(0));    // Visualizzazione del nodo
    return root;
}

/**
 * Fusione delle regioni adiacenti.
 * A partire dai quattro sotto-nodi:
 * - confronta le medie e fonde le regioni con differenza inferiore a mTH
 * - gestisce fusioni lineari (i e i+1), ma anche possibili terzi aggregati
 * - le regioni non fuse vengono processate ricorsivamente
 */
void merge(TNode *root) {
    if (root->region.width > tSize && root->stddev > smTH) {

        int mean[4];
        for (int i = 0; i < 4; i++)
            mean[i] = (int) root->regions[i]->mean;

        // Tentativi di fusione tra regioni adiacenti
        for (int i = 0; i < 4; i++) {
            int next = (i + 1) % 4;

            if (abs(mean[i] - mean[next]) < mTH) {
                root->merged.push_back(root->regions[i]);
                root->merged.push_back(root->regions[next]);
                root->isMerged[i] = root->isMerged[next] = true;

                // Possibile ulteriore fusione
                int next2 = (i + 2) % 4;
                int prev = (i + 3) % 4;

                if (abs(mean[next] - mean[next2]) < mTH) {
                    root->merged.push_back(root->regions[next2]);
                    root->isMerged[next2] = true;
                } else if (abs(mean[prev] - mean[i]) < mTH) {
                    root->merged.push_back(root->regions[prev]);
                    root->isMerged[prev] = true;
                }
            }
        }

        // Ricorsione sulle regioni non fuse
        for (int i = 0; i < 4; i++)
            if (!root->isMerged[i])
                merge(root->regions[i]);

    } else {
        // Regione già omogenea o troppo piccola
        root->merged.push_back(root);
    }
}

/**
 * Segmentazione finale.
 * Per ogni gruppo di regioni fuse:
 * - calcola la media delle medie
 * - assegna tale valore all’intera area
 * Procede ricorsivamente sulle regioni non fuse.
 */
void segment(TNode *root, Mat &img) {
    float val = 0;

    for (auto node : root->merged)
        val += node->mean;

    val /= root->merged.size();

    // Riempimento delle regioni fuse
    for (auto node : root->merged)
        img(node->region) = (int) val;

    // Ricorsione sulle regioni restanti
    for (int i = 0; i < 4; i++)
        if (!root->isMerged[i] && root->regions[i])
            segment(root->regions[i], img);
}

/**
 * Funzione principale dell’algoritmo:
 * - Clona l’immagine
 * - Applica un Gaussian Blur per ridurre il rumore
 * - Limita l’immagine a una dimensione potenza di 2
 * - Esegue lo split, merge e segmentazione
 * - Mostra la struttura del Quad-Tree e l’immagine segmentata
 */
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
