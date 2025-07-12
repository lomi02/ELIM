#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// Parametri globali di soglia
double smThreshold = 10;   // Soglia di deviazione standard per il criterio di split
int minRegSize = 8;        // Dimensione minima delle regioni
int mThreshold = 1;        // Soglia di differenza di intensità per il merge

/**
 * Classe che rappresenta un nodo dell'albero quaternario.
 *
 * Ogni nodo contiene:
 * - la regione rettangolare associata
 * - i 4 figli (regioni suddivise)
 * - l'elenco delle regioni unite (merge)
 * - indicatori booleani per i figli uniti
 * - le statistiche (media e deviazione standard)
 */
class TNode {
public:
    Rect region;                       // Regione rettangolare del nodo
    vector<TNode *> regions = vector<TNode *>(4, nullptr); // Figli
    vector<TNode *> merged;           // Regioni unite (merge)
    vector<bool> isMerged = vector(4, false); // Flag per tenere traccia dei figli uniti
    double stddev, mean, meanMerged;  // Statistiche

    TNode(Rect R) { region = R; }
};

/**
 * Funzione ricorsiva per la fase di SPLIT.
 *
 * Divide ricorsivamente la regione R in 4 sotto-regioni
 * se la deviazione standard supera la soglia e se la dimensione è sufficiente.
 * Inoltre, disegna il rettangolo della regione per visualizzare il quadtree.
 *
 * @param src  Immagine da segmentare
 * @param R    Regione corrente da analizzare
 * @return     Puntatore al nodo creato
 */
TNode *split(Mat &src, Rect R) {
    auto node = new TNode(R);

    // Calcolo della media e deviazione standard della regione
    Scalar stddev, mean;
    meanStdDev(src(R), mean, stddev);
    node->stddev = stddev[0];
    node->mean = mean[0];

    // Condizione per lo split: dimensione minima e deviazione elevata
    if (R.width > minRegSize && node->stddev > smThreshold) {

        // Suddivide la regione in 4 sotto-regioni (quadtree)
        node->regions[0] = split(src, Rect(R.x, R.y, R.height / 2, R.width / 2));
        node->regions[1] = split(src, Rect(R.x, R.y + R.width / 2, R.height / 2, R.width / 2));
        node->regions[2] = split(src, Rect(R.x + R.height / 2, R.y, R.height / 2, R.width / 2));
        node->regions[3] = split(src, Rect(R.x + R.height / 2, R.y + R.width / 2, R.height / 2, R.width / 2));
    }

    // Disegna il bordo della regione sul risultato
    rectangle(src, R, Scalar(0));
    return node;
}

/**
 * Funzione ricorsiva per la fase di MERGE.
 *
 * Unisce le regioni figlie adiacenti se la differenza tra le loro medie
 * di intensità è inferiore alla soglia mThreshold.
 *
 * @param node Nodo corrente su cui applicare il merge
 */
void merge(TNode *node) {

    // Applica merge solo se ha figli e deviazione sufficiente
    if (node->region.width > minRegSize && node->stddev > smThreshold) {
        for (int i = 0; i < 4; i++) {

            // Verifica la somiglianza con il vicino (i+1)
            if (abs((int) node->regions[i]->mean - (int) node->regions[(i + 1) % 4]->mean) < mThreshold) {
                node->merged.push_back(node->regions[i]);
                node->isMerged[i] = true;

                node->merged.push_back(node->regions[(i + 1) % 4]);
                node->isMerged[(i + 1) % 4] = true;

                // Verifica possibilità di unione anche col terzo vicino
                if (abs((int) node->regions[(i + 1) % 4]->mean - (int) node->regions[(i + 2) % 4]->mean) < mThreshold) {
                    node->merged.push_back(node->regions[(i + 2) % 4]);
                    node->isMerged[(i + 2) % 4] = true;
                    break;
                }

                // Oppure con il quarto vicino
                if (abs((int) node->regions[(i + 3) % 4]->mean - (int) node->regions[i]->mean) < mThreshold) {
                    node->merged.push_back(node->regions[(i + 3) % 4]);
                    node->isMerged[(i + 3) % 4] = true;
                    break;
                }
            }
        }

        // Applica merge ricorsivamente ai sotto-nodi non uniti
        for (int i = 0; i < 4; i++)
            if (!node->isMerged[i])
                merge(node->regions[i]);
    } else {

        // Se non è possibile suddividere, considera il nodo come regione finale
        node->merged.push_back(node);
    }
}

/**
 * Funzione ricorsiva per la fase di SEGMENTAZIONE.
 *
 * Applica l'intensità media calcolata alle regioni finali unite.
 *
 * @param src Nodo corrente
 * @param dst Immagine risultante da segmentare
 */
void segment(TNode *src, Mat &dst) {
    float val = 0;

    // Calcola la media delle regioni unite
    for (auto node: src->merged)
        val += node->mean;
    val /= src->merged.size();

    // Applica la media alla regione
    for (auto node: src->merged)
        dst(node->region) = (int) val;

    // Segmenta ricorsivamente i figli non uniti
    for (int i = 0; i < 4; i++)
        if (!src->isMerged[i] && src->regions[i])
            segment(src->regions[i], dst);
}

/**
 * Funzione principale per eseguire l’intero algoritmo Split and Merge.
 *
 * Passaggi:
 * 1. Preprocessing con sfocatura (GaussianBlur)
 * 2. SPLIT: divisione ricorsiva delle regioni
 * 3. MERGE: unione delle regioni simili
 * 4. SEGMENT: assegnazione delle intensità alle regioni finali
 *
 * @param src     Immagine originale in input (grayscale)
 * @param output  Immagine risultante segmentata
 */
void splitAndMerge(Mat &src, Mat &output) {

    // Ridimensiona l’immagine a potenza di 2 per facilitare il quadtree
    int exponent = log(min(src.rows, src.cols)) / log(2);
    int size = pow(2.0, double(exponent));
    Mat img = src(Rect(0, 0, size, size)).clone();

    // Preprocessing: sfocatura per ridurre il rumore
    GaussianBlur(img, img, Size(3, 3), 0, 0);

    // Fase SPLIT: costruzione del quadtree
    TNode *node = split(img, Rect(0, 0, img.rows, img.cols));

    // Fase MERGE: unione delle regioni adiacenti simili
    merge(node);

    // Copia dell’immagine originale per la segmentazione
    output = src(Rect(0, 0, size, size)).clone();

    // Fase SEGMENT: applicazione delle intensità
    segment(node, output);

    // Visualizzazione dei risultati
    imshow("Working Image", img);       // Visualizza il quadtree disegnato
    imshow("Output Image", output);     // Visualizza immagine segmentata
    waitKey(0);
}