#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

/**
 * Algoritmo Split and Merge per segmentazione di immagini in scala di grigi.
 *
 * L'algoritmo divide l'immagine in quadranti fino a soddisfare un criterio di omogeneità,
 * quindi tenta di unire regioni adiacenti con caratteristiche simili.
 * La segmentazione finale assegna a ciascuna regione un valore medio di intensità.
 *
 * Parametri globali:
 * smThreshold : soglia di deviazione standard per decidere se dividere
 * minRegSize  : dimensione minima della regione (in pixel)
 * mThreshold  : soglia massima di differenza media tra regioni per consentire la fusione
 */
double smThreshold = 10; // Soglia deviazione standard per split
int minRegSize = 8;      // Dimensione minima regione per split
int mThreshold = 1;      // Differenza massima tra medie per merge

/**
 * Nodo dell'albero di segmentazione.
 * Contiene la regione rettangolare, i sotto-nodi (suddivisione), informazioni sulla fusione
 * e valori statistici (media e deviazione standard) della regione.
 */
class TNode {
public:
    Rect region;                  // Rettangolo della regione
    vector<TNode *> regions;      // Sotto-regioni (4 quadranti)
    vector<TNode *> merged;       // Regioni effettivamente fuse
    vector<bool> isMerged;        // Flag di fusione per ogni sotto-regione
    double stddev, mean;          // Statistiche della regione

    TNode(Rect R) : region(R), regions(4, nullptr), isMerged(4, false) {}

    // Decide se la regione deve essere ulteriormente divisa
    bool shouldSplit() const {
        return region.width > minRegSize && stddev > smThreshold;
    }
};

/**
 * Suddivide ricorsivamente la regione in quadranti.
 * Calcola la deviazione standard e media di ciascuna regione.
 */
TNode *split(Mat &src, Rect R) {
    TNode *node = new TNode(R);
    Scalar stddev, mean;
    meanStdDev(src(R), mean, stddev); // Calcola media e deviazione standard
    node->stddev = stddev[0];
    node->mean = mean[0];

    if (node->shouldSplit()) {
        int halfH = R.height / 2;
        int halfW = R.width / 2;
        // Suddivisione in 4 quadranti
        node->regions[0] = split(src, Rect(R.x, R.y, halfH, halfW));
        node->regions[1] = split(src, Rect(R.x, R.y + halfW, halfH, halfW));
        node->regions[2] = split(src, Rect(R.x + halfH, R.y, halfH, halfW));
        node->regions[3] = split(src, Rect(R.x + halfH, R.y + halfW, halfH, halfW));
    }

    rectangle(src, R, Scalar(0)); // Disegna la regione sull'immagine di lavoro
    return node;
}

/**
 * Controlla se due regioni possono essere fuse in base alla differenza media.
 */
bool canMerge(TNode *a, TNode *b) {
    return abs((int)a->mean - (int)b->mean) < mThreshold;
}

/**
 * Fusione ricorsiva delle regioni.
 * Se due o più regioni adiacenti hanno medie simili, vengono fuse.
 */
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

/**
 * Assegna a ciascuna regione il valore medio dopo la fusione.
 */
void segment(TNode *node, Mat &dst) {
    float val = 0;
    for (size_t i = 0; i < node->merged.size(); i++)
        val += node->merged[i]->mean;
    val /= node->merged.size();

    for (size_t i = 0; i < node->merged.size(); i++)
        dst(node->merged[i]->region) = (int)val;

    for (int i = 0; i < 4; i++)
        if (!node->isMerged[i] && node->regions[i])
            segment(node->regions[i], dst);
}

/**
 * Funzione principale che coordina il processo di Split and Merge.
 * Applica il Gaussian Blur, esegue la suddivisione, la fusione e la segmentazione finale.
 */
void splitAndMerge(Mat &src, Mat &output, Mat &working) {
    int exponent = log(min(src.rows, src.cols)) / log(2);
    int size = pow(2.0, (double)exponent); // Ridimensionamento a una potenza di 2
    working = src(Rect(0, 0, size, size)).clone();

    GaussianBlur(working, working, Size(3, 3), 0, 0);
    TNode *node = split(working, Rect(0, 0, working.rows, working.cols));
    merge(node);

    output = src(Rect(0, 0, size, size)).clone();
    segment(node, output);
}
