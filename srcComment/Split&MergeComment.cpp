#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// Struttura nodo per l'albero quaternario
// Ogni nodo rappresenta una regione rettangolare dell'immagine
struct QNode {
    Rect rect;                      // Regione rettangolare del nodo
    QNode *child[4] = {nullptr};    // 4 figli per il quadtree
    double mean, dev;               // Media e deviazione standard della regione
    bool isLeaf = true;             // Flag per identificare le foglie

    QNode(Rect r) : rect(r) {}
};

/**
 * Funzione ricorsiva per la fase di SPLIT dell'algoritmo Split and Merge.
 *
 * Divide ricorsivamente l'immagine in quadranti se:
 * - La dimensione della regione è maggiore della dimensione minima
 * - La deviazione standard è maggiore della soglia (regione non omogenea)
 *
 * @param img       Immagine su cui disegnare i bordi del quadtree
 * @param rect      Regione rettangolare corrente da analizzare
 * @param splitTH   Soglia di deviazione standard per decidere se dividere
 * @param minSize   Dimensione minima sotto la quale non dividere
 *
 * @return Puntatore al nodo radice della regione
 */
QNode *split(Mat &img, Rect rect, double splitTH, int minSize) {
    auto node = new QNode(rect);

    // Calcolo delle statistiche della regione corrente
    // mean = intensità media dei pixel nella regione
    // dev = deviazione standard (misura di omogeneità)
    Scalar mean, dev;
    meanStdDev(img(rect), mean, dev);
    node->mean = mean[0];
    node->dev = dev[0];

    // Condizione di split: dimensione sufficiente E regione non omogenea
    if (rect.width > minSize && node->dev > splitTH) {
        node->isLeaf = false;
        int halfW = rect.width / 2, halfH = rect.height / 2;

        // Creazione dei 4 quadranti (ricorsivamente)
        // Quadrante 0: top-left
        node->child[0] = split(img, Rect(rect.x, rect.y, halfW, halfH), splitTH, minSize);

        // Quadrante 1: top-right
        node->child[1] = split(img, Rect(rect.x + halfW, rect.y, halfW, halfH), splitTH, minSize);

        // Quadrante 2: bottom-left
        node->child[2] = split(img, Rect(rect.x, rect.y + halfH, halfW, halfH), splitTH, minSize);

        // Quadrante 3: bottom-right
        node->child[3] = split(img, Rect(rect.x + halfW, rect.y + halfH, halfW, halfH), splitTH, minSize);
    }

    // Disegna il bordo della regione per visualizzare il quadtree
    rectangle(img, rect, Scalar(0));
    return node;
}

/**
 * Funzione ricorsiva per la fase di MERGE dell'algoritmo Split and Merge.
 *
 * Unisce regioni adiacenti che hanno intensità simile:
 * - Processa prima ricorsivamente tutti i figli
 * - Verifica se tutti i figli sono foglie con intensità simile
 * - Se sì, unisce i figli trasformando il nodo corrente in foglia
 *
 * @param node      Nodo corrente da processare
 * @param mergeTH   Soglia di differenza di intensità per il merge
 */
void merge(QNode *node, double mergeTH) {
    if (node->isLeaf) return;

    // Passo 1: Ricorsione su tutti i figli (bottom-up)
    for (int i = 0; i < 4; i++)
        merge(node->child[i], mergeTH);

    // Passo 2: Verifica se è possibile fare il merge
    // Condizioni: tutti i figli devono essere foglie con intensità simile
    bool canMerge = true;
    for (int i = 0; i < 4; i++) {

        // Se un figlio non è foglia, non possiamo fare merge
        if (!node->child[i]->isLeaf) {
            canMerge = false;
            break;
        }
        // Se la differenza di intensità è troppo grande, non fare merge
        if (abs(node->child[i]->mean - node->child[0]->mean) > mergeTH) {
            canMerge = false;
            break;
        }
    }

    // Passo 3: Esegui il merge se possibile
    if (canMerge) {
        node->isLeaf = true;

        // Calcola la nuova intensità media come media dei figli
        node->mean = 0;
        for (int i = 0; i < 4; i++)
            node->mean += node->child[i]->mean;
        node->mean /= 4;
    }
}

/**
 * Funzione ricorsiva per la fase di SEGMENTAZIONE.
 *
 * Applica l'intensità media calcolata a tutte le regioni finali:
 * - Se il nodo è una foglia, colora tutta la regione con la sua intensità media
 * - Altrimenti, processa ricorsivamente tutti i figli
 *
 * @param node  Nodo corrente da segmentare
 * @param img   Immagine su cui applicare la segmentazione
 */
void segment(QNode *node, Mat &img) {
    if (node->isLeaf)
        // Colora tutta la regione con l'intensità media
        img(node->rect) = (int) node->mean;
    else
        // Ricorsione sui figli
        for (int i = 0; i < 4; i++)
            segment(node->child[i], img);
}

/**
 * Implementazione completa dell'algoritmo Split and Merge per segmentazione di immagini.
 *
 * L'algoritmo esegue tre fasi principali:
 * 1. SPLIT: Divide l'immagine in regioni basandosi sulla omogeneità
 * 2. MERGE: Unisce regioni adiacenti con caratteristiche simili
 * 3. SEGMENT: Applica un'intensità uniforme a ogni regione finale
 *
 * @param img       Immagine in input (scala di grigi)
 * @param splitTH   Soglia per la fase di split (omogeneità)
 * @param minSize   Dimensione minima delle regioni
 * @param mergeTH   Soglia per la fase di merge (similarità)
 */
void splitAndMerge(Mat img, double splitTH, int minSize, double mergeTH) {

    // Preprocessing: smoothing per ridurre il rumore
    GaussianBlur(img, img, Size(5, 5), 0);

    // Ridimensionamento a quadrato perfetto (potenza di 2)
    // Necessario per la struttura ricorsiva del quadtree
    int exponent = log(min(img.cols, img.rows)) / log(2);
    int size = pow(2.0, (double) exponent);
    img = img(Rect(0, 0, size, size)).clone();

    // Creazione delle immagini di output
    Mat qtree = img.clone();        // Per visualizzare il quadtree
    Mat segmented = img.clone();    // Per la segmentazione finale

    // Fase 1: SPLIT - Costruzione del quadtree
    QNode *root = split(qtree, Rect(0, 0, size, size), splitTH, minSize);

    // Fase 2: MERGE - Unione delle regioni simili
    merge(root, mergeTH);

    // Fase 3: SEGMENT - Applicazione della segmentazione
    segment(root, segmented);

    // Visualizzazione dei risultati
    imshow("Quad Tree", qtree);         // Mostra la struttura del quadtree
    imshow("Segmented", segmented);     // Mostra l'immagine segmentata
    waitKey(0);
}