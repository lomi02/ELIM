#include <opencv2/opencv.hpp>
#include <stack>
using namespace cv;
using namespace std;

/**
 * Applica l'algoritmo di Region Growing per segmentare un'immagine in scala di grigi.
 *
 * Questa funzione esegue i seguenti passaggi:
 * 1. Scorre tutti i pixel dell'immagine come semi iniziali
 * 2. Per ogni seed non ancora etichettato, espande la regione includendo i vicini simili
 * 3. La similarità è basata sulla differenza assoluta dei valori di intensità
 * 4. Se la regione è abbastanza grande, le assegna un'etichetta unica; altrimenti viene etichettata come rumore
 * 5. Restituisce una mappa di etichette con tutte le regioni segmentate
 *
 * @param input  Immagine in input (scala di grigi)
 *
 * @return Matrice di etichette (CV_8U) dove ogni regione ha un valore unico
 */
Mat regionGrowing(Mat &input) {
    Mat src = input.clone();

    // Parametri dell'algoritmo
    int similarityThreshold = 5;       // Differenza massima tra pixel per essere considerati simili
    double minAreaFactor = 0.01;       // Frazione minima di area dell'immagine per accettare la regione
    uchar maxLabels = 100;             // Numero massimo di etichette

    int minArea = int(minAreaFactor * src.rows * src.cols);     // Calcolo area minima
    Mat labels = Mat::zeros(src.rows, src.cols, CV_8U);         // Matrice di etichette inizializzata a 0
    Mat regionMask = Mat::zeros(src.rows, src.cols, CV_8U);     // Maschera temporanea della regione
    uchar currentLabel = 1;                                     // Etichetta corrente

    // 8 vicini (connessione 8) per esplorazione dei pixel adiacenti
    const Point neighbors[8] = {
        Point(1, 0), Point(1, -1), Point(0, -1), Point(-1, -1),
        Point(-1, 0), Point(-1, 1), Point(0, 1), Point(1, 1)
    };

    // Passo 1: Scansione di tutti i pixel dell'immagine
    for (int x = 0; x < src.cols; x++) {
        for (int y = 0; y < src.rows; y++) {
            Point seed(x, y);
            if (labels.at<uchar>(seed) != 0) continue; // Salta pixel già etichettati

            // Stack per l'espansione della regione
            stack<Point> points;
            points.push(seed);
            regionMask.setTo(0); // Reset maschera regione

            // Passo 2: Espansione della regione usando Region Growing
            while (!points.empty()) {
                Point p = points.top();
                points.pop();
                regionMask.at<uchar>(p) = 1;
                uchar centerVal = src.at<uchar>(p);

                // Passo 3: Controllo dei pixel vicini
                for (int i = 0; i < 8; i++) {
                    Point q = p + neighbors[i];
                    if (q.x < 0 || q.x >= src.cols || q.y < 0 || q.y >= src.rows)
                        continue; // Salta pixel fuori immagine
                    if (labels.at<uchar>(q) || regionMask.at<uchar>(q))
                        continue; // Salta pixel già etichettati o già nella regione
                    uchar neighVal = src.at<uchar>(q);
                    if (abs(int(centerVal) - int(neighVal)) < similarityThreshold) {
                        regionMask.at<uchar>(q) = 1;
                        points.push(q); // Aggiunge alla regione
                    }
                }
            }

            // Passo 4: Verifica della dimensione della regione
            int regionArea = int(sum(regionMask)[0]);
            if (regionArea > minArea) {
                labels += regionMask * currentLabel; // Assegna etichetta unica
                if (++currentLabel > maxLabels) return labels; // Limite di etichette
            } else {
                labels += regionMask * 255; // Region considered noise
            }
        }
    }

    return labels;
}
