#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
 * Parametri globali per l’algoritmo Split & Merge:
 *
 * smTH : soglia sulla deviazione standard per decidere quando suddividere una regione
 * tSize: dimensione minima consentita per effettuare ulteriori suddivisioni
 * mTH  : soglia massima di differenza tra le medie che consente la fusione di due regioni
 */
double smTH = 10;
int tSize = 8;
int mTH = 5;

/**
 * Nodo dell’albero di suddivisione (Quad-Tree).
 * Ogni nodo conserva:
 * - La regione rettangolare corrispondente
 * - I quattro sotto-nodi generati dalla suddivisione (se avviene)
 * - L’elenco delle regioni fuse
 * - Un vettore di flag che indica quali sotto-regioni sono state fuse
 * - Le statistiche locali: media, deviazione standard e media dopo fusione
 */
class TNode {
public:
    Rect region;                                            // Regione associata al nodo
    vector<TNode *> regions = vector<TNode *>(4, nullptr);  // Sotto-regioni (4 quadranti)
    vector<TNode *> merged;                                 // Regioni fuse
    vector<bool> isMerged = vector<bool>(4, false);         // Flag di fusione
    double stddev, mean, meanMerged;                        // Statistiche della regione

    TNode(Rect R) { region = R; }
};

/**
 * Suddivide ricorsivamente la regione in quadranti.
 * Per ogni nodo:
 * - calcola media e deviazione standard dell’area
 * - se la deviazione supera la soglia e la regione è abbastanza grande → suddivide
 * - disegna il contorno della regione per visualizzare la struttura del Quad-Tree
 */
TNode *split(Mat &img, Rect R) {
    TNode *root = new TNode(R);

    Scalar stddev, mean;
    meanStdDev(img(R), mean, stddev); // Calcolo statistiche locali

    root->mean = mean[0];
    root->stddev = stddev[0];

    // Condizione di suddivisione
    if (R.width > tSize && root->stddev > smTH) {
        root->regions[0] = split(img, Rect(R.x, R.y, R.height / 2, R.width / 2));
        root->regions[1] = split(img, Rect(R.x, R.y + R.width / 2, R.height / 2, R.width / 2));
        root->regions[2] = split(img, Rect(R.x + R.height / 2, R.y, R.height / 2, R.width / 2));
        root->regions[3] = split(img, Rect(R.x + R.height / 2, R.y + R.width / 2, R.height / 2, R.width / 2));
    }

    rectangle(img, R, Scalar(0)); // Visualizzazione del nodo
    return root;
}

/**
 * Fusione delle regioni adiacenti.
 * Verifica se le medie di regioni contigue differiscono meno della soglia mTH.
 * In tal caso, le regioni vengono unite nello stesso gruppo.
 * Le regioni non fuse vengono esplorate ricorsivamente.
 */
void merge(TNode *root) {
    if (root->region.width > tSize && root->stddev > smTH) {
        for (int i = 0; i < 4; i++) {

            // Controllo fusione con regione successiva (in modulo 4)
            if (abs((int) root->regions[i]->mean - (int) root->regions[(i + 1) % 4]->mean) < mTH) {
                root->merged.push_back(root->regions[i]);
                root->isMerged[i] = true;

                root->merged.push_back(root->regions[(i + 1) % 4]);
                root->isMerged[(i + 1) % 4] = true;

                // Possibile fusione anche con il terzo quadrante consecutivo
                if (abs((int) root->regions[(i + 1) % 4]->mean - (int) root->regions[(i + 2) % 4]->mean) < mTH) {
                    root->merged.push_back(root->regions[(i + 2) % 4]);
                    root->isMerged[(i + 2) % 4] = true;
                    break;
                }

                // Oppure con il quadrante precedente
                if (abs((int) root->regions[(i + 3) % 4]->mean - (int) root->regions[i]->mean) < mTH) {
                    root->merged.push_back(root->regions[(i + 3) % 4]);
                    root->isMerged[(i + 3) % 4] = true;
                    break;
                }
            }
        }

        // Ricorsione sulle regioni non fuse
        for (int i = 0; i < 4; i++)
            if (!root->isMerged[i])
                merge(root->regions[i]);
    } else {

        // Regione troppo piccola o già omogenea → nodo come singola unità
        root->merged.push_back(root);
    }
}

/**
 * Assegna a ciascun gruppo di regioni fuse un valore medio uniforme.
 * Dopo la fusione:
 * - si calcola la media delle regioni unite
 * - l’intera area viene riempita con quel valore
 * Procede ricorsivamente sulle regioni non fuse.
 */
void segment(TNode *root, Mat &img) {
    float val = 0;

    for (auto node: root->merged)
        val += node->mean;

    val /= root->merged.size(); // Media del gruppo fuso

    // Riempimento dell’area con il valore medio
    for (auto node: root->merged)
        img(node->region) = (int) val;

    // Ricorsione sulle regioni non fuse
    for (int i = 0; i < 4; i++) {
        if (!root->isMerged[i] && root->regions[i])
            segment(root->regions[i], img);
    }
}

/**
 * Funzione principale:
 * - Clona l’immagine originale
 * - Applica un leggero Gaussian Blur per ridurre il rumore
 * - Ridimensiona a una dimensione potenza di 2 (necessario per Quad-Tree regolare)
 * - Esegue split, merge e segmentazione finale
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
