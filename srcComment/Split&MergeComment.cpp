#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
 * Parametri dell'algoritmo Split & Merge:
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
 * Rappresenta una regione dell'immagine e include:
 * - Il rettangolo della regione
 * - I quattro eventuali sotto-nodi risultanti dalla suddivisione
 * - L'elenco delle regioni fuse
 * - Flag che indicano quali sotto-regioni sono state fuse
 * - Le statistiche della regione (media e deviazione standard)
 */
class TNode {
public:
    Rect region;                    // Rettangolo che delimita la regione
    TNode *regions[4] = {nullptr};  // Array dei 4 sotto-nodi (quadranti)
    vector<TNode *> merged;         // Vettore delle regioni che sono state fuse insieme
    bool isMerged[4] = {false};     // Flag per indicare se ciascun sotto-nodo è stato fuso
    double stddev, mean;            // Deviazione standard e media dei pixel nella regione

    // Costruttore: inizializza la regione
    TNode(Rect R) : region(R) {}
};

/**
 * Fase di SPLIT: suddivide ricorsivamente la regione in 4 quadranti.
 *
 * Algoritmo:
 * 1. Crea un nodo per la regione corrente
 * 2. Calcola media e deviazione standard dei pixel nella regione
 * 3. Se la regione è sufficientemente grande E non omogenea:
 *    - Divide la regione in 4 quadranti
 *    - Chiama ricorsivamente split su ciascun quadrante
 * 4. Disegna il bordo della regione per visualizzazione
 * 5. Restituisce il nodo creato
 */
TNode *split(Mat &img, Rect R) {
    // Crea un nuovo nodo per questa regione
    TNode *root = new TNode(R);

    // Calcola le statistiche (media e deviazione standard) per la regione
    Scalar stddev, mean;
    meanStdDev(img(R), mean, stddev);  // OpenCV calcola entrambe in una volta
    root->mean = mean[0];              // Estrai il valore scalare della media
    root->stddev = stddev[0];          // Estrai il valore scalare della dev. std.

    // Verifica se dobbiamo continuare a dividere:
    // - La regione deve essere più grande della dimensione minima (tSize)
    // - La deviazione standard deve superare la soglia (smTH)
    //   Una dev. std. alta indica che la regione non è omogenea
    if (R.width > tSize && root->stddev > smTH) {
        // Calcola le dimensioni dei quadranti (metà altezza e metà larghezza)
        int h = R.height / 2;
        int w = R.width / 2;

        // Crea i 4 quadranti ricorsivamente:
        // Quadrante 0: in alto a sinistra
        root->regions[0] = split(img, Rect(R.x, R.y, h, w));

        // Quadrante 1: in alto a destra
        root->regions[1] = split(img, Rect(R.x, R.y + w, h, w));

        // Quadrante 2: in basso a sinistra
        root->regions[2] = split(img, Rect(R.x + h, R.y, h, w));

        // Quadrante 3: in basso a destra
        root->regions[3] = split(img, Rect(R.x + h, R.y + w, h, w));
    }
    // Se non dividiamo, questo nodo diventa una foglia dell'albero

    // Disegna il bordo della regione per visualizzare la struttura del Quad-Tree
    rectangle(img, R, Scalar(0));

    return root;
}

/**
 * Fase di MERGE: fonde le regioni adiacenti che hanno caratteristiche simili.
 *
 * Algoritmo:
 * 1. Se il nodo è stato suddiviso (ha 4 sotto-regioni):
 *    a. Pre-calcola le medie convertite a int per evitare cast ripetuti
 *    b. Per ogni coppia di regioni adiacenti (i, i+1):
 *       - Se la differenza tra le loro medie < mTH → le fonde
 *       - Tenta di aggiungere anche una terza regione al gruppo
 *    c. Chiama ricorsivamente merge sulle regioni non fuse
 * 2. Se il nodo è una foglia (non suddiviso):
 *    - Aggiunge se stesso come regione da segmentare
 */
void merge(TNode *root) {
    // Verifica se questo nodo è stato suddiviso
    // (stesse condizioni usate durante lo split)
    if (root->region.width > tSize && root->stddev > smTH) {

        // Pre-calcola le medie come interi per evitare cast multipli
        // Questo rende i confronti più efficienti
        int mean[4];
        for (int i = 0; i < 4; i++)
            mean[i] = (int) root->regions[i]->mean;

        // Esamina tutte le possibili coppie di regioni adiacenti
        // Le regioni sono disposte in senso circolare: 0→1→2→3→0
        for (int i = 0; i < 4; i++) {
            int next = (i + 1) % 4;  // Regione successiva (con wrap-around)

            // Se le medie di due regioni adiacenti sono simili:
            // (differenza assoluta < mTH)
            if (abs(mean[i] - mean[next]) < mTH) {
                // Aggiungi entrambe le regioni al gruppo di fusione
                root->merged.push_back(root->regions[i]);
                root->merged.push_back(root->regions[next]);

                // Segna entrambe come fuse
                root->isMerged[i] = root->isMerged[next] = true;

                // Tenta di espandere il gruppo con una terza regione
                int next2 = (i + 2) % 4;  // Due posizioni avanti
                int prev = (i + 3) % 4;   // Una posizione indietro (equivalente a i-1)

                // Caso 1: la regione next2 è simile a next
                if (abs(mean[next] - mean[next2]) < mTH) {
                    root->merged.push_back(root->regions[next2]);
                    root->isMerged[next2] = true;
                }
                // Caso 2: la regione prev è simile a i
                else if (abs(mean[prev] - mean[i]) < mTH) {
                    root->merged.push_back(root->regions[prev]);
                    root->isMerged[prev] = true;
                }
                // Nota: usiamo else if perché vogliamo aggiungere al massimo
                // una terza regione per gruppo
            }
        }

        // Processa ricorsivamente le sotto-regioni che NON sono state fuse
        // Queste potrebbero avere le loro proprie fusioni a livelli più profondi
        for (int i = 0; i < 4; i++)
            if (!root->isMerged[i])
                merge(root->regions[i]);

    } else {
        // Questo nodo è una foglia (regione omogenea o troppo piccola)
        // Aggiungila direttamente come regione da segmentare
        root->merged.push_back(root);
    }
}

/**
 * Fase di SEGMENTAZIONE: assegna a ogni gruppo di regioni fuse un valore uniforme.
 *
 * Algoritmo:
 * 1. Per il gruppo di regioni fuse in questo nodo:
 *    a. Calcola la media delle medie di tutte le regioni nel gruppo
 *    b. Assegna questo valore uniforme a tutti i pixel di tutte le regioni
 * 2. Chiama ricorsivamente segment sulle sotto-regioni non fuse
 */
void segment(TNode *root, Mat &img) {
    // Calcola la media del gruppo di regioni fuse
    float val = 0;

    // Somma tutte le medie delle regioni nel gruppo
    for (auto node : root->merged)
        val += node->mean;

    // Dividi per il numero di regioni per ottenere la media complessiva
    val /= root->merged.size();

    // Assegna il valore medio uniforme a tutte le regioni del gruppo
    // Questo crea l'effetto di segmentazione visibile nell'immagine finale
    for (auto node : root->merged)
        img(node->region) = (int) val;

    // Processa ricorsivamente le sotto-regioni che non sono state fuse
    // a questo livello (potrebbero avere le loro fusioni a livelli più profondi)
    for (int i = 0; i < 4; i++)
        if (!root->isMerged[i] && root->regions[i])
            segment(root->regions[i], img);
}

/**
 * Funzione principale dell'algoritmo Split & Merge.
 *
 * Passi:
 * 1. Preprocessing dell'immagine:
 *    - Clona l'immagine originale per non modificarla
 *    - Applica Gaussian Blur per ridurre il rumore
 *    - Ritaglia l'immagine a una dimensione che sia potenza di 2
 *      (necessario per garantire suddivisioni simmetriche nel Quad-Tree)
 * 2. Esegue le tre fasi dell'algoritmo:
 *    - Split: costruisce il Quad-Tree
 *    - Merge: identifica regioni simili da fondere
 *    - Segment: applica la segmentazione finale
 * 3. Mostra i risultati
 */
void SplitMerge(Mat &input) {
    // Clona l'immagine per non modificare l'originale
    Mat img = input.clone();

    // Applica filtro Gaussiano per ridurre il rumore
    // Kernel 3x3, sigma = 1 per entrambe le direzioni
    GaussianBlur(img, img, Size(3, 3), 1, 1);

    // Calcola la massima potenza di 2 contenuta nelle dimensioni dell'immagine
    // Esempio: se l'immagine è 500x600, min(500,600)=500
    //          log2(500) ≈ 8.97, quindi exponent = 8
    //          2^8 = 256, quindi useremo una regione 256x256
    int exponent = log(min(img.cols, img.rows)) / log(2);
    int quadSize = pow(2.0, (double) exponent);

    // Estrae un quadrato di dimensione potenza di 2 dall'angolo in alto a sinistra
    Rect square = Rect(0, 0, quadSize, quadSize);
    img = img(square).clone();

    // Crea una copia per l'immagine segmentata
    Mat imgSeg = img.clone();

    // FASE 1: SPLIT
    // Costruisce il Quad-Tree dividendo ricorsivamente l'immagine
    // Nota: passiamo (rows, cols) perché Rect vuole (height, width)
    TNode *root = split(img, Rect(0, 0, img.rows, img.cols));

    // FASE 2: MERGE
    // Identifica e raggruppa le regioni con caratteristiche simili
    merge(root);

    // FASE 3: SEGMENT
    // Applica la segmentazione finale assegnando valori uniformi ai gruppi
    segment(root, imgSeg);

    // Mostra i risultati:
    // - "Quad Tree": visualizza la struttura gerarchica delle suddivisioni
    // - "Segmented": immagine finale segmentata con regioni omogenee
    imshow("Quad Tree", img);
    imshow("Segmented", imgSeg);
    waitKey(0);
}