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
    Mat img = input.clone();

    // Parametri dell'algoritmo
    int simTH = 5;       // Differenza massima tra pixel per essere considerati simili
    double minAreaFactor = 0.01;       // Frazione minima di area dell'immagine per accettare la regione
    uchar maxLabels = 100;             // Numero massimo di etichette

    int minArea = int(minAreaFactor * img.rows * img.cols);     // Calcolo area minima
    Mat labels = Mat::zeros(img.rows, img.cols, CV_8U);         // Matrice di etichette inizializzata a 0
    Mat regionMask = Mat::zeros(img.rows, img.cols, CV_8U);     // Maschera temporanea della regione
    uchar currentLabel = 1;                                     // Etichetta corrente

    // 8 vicini (connessione 8) per esplorazione dei pixel adiacenti
    const Point neighbors[8] = {
        Point(1, 0), Point(1, -1), Point(0, -1), Point(-1, -1),
        Point(-1, 0), Point(-1, 1), Point(0, 1), Point(1, 1)
    };

    // Passo 1: Scansione di tutti i pixel dell'immagine
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++) {

            Point seed(y, x);

            // Salta pixel già etichettati
            if (labels.at<uchar>(seed) != 0)
                continue;

            // Stack per l'espansione della regione di punti da processare
            stack<Point> points;
            points.push(seed);

            // Reset maschera regione
            regionMask.setTo(0);

            // Passo 2: Espansione della regione usando Region Growing
            while (!points.empty()) {
                Point current = points.top();
                points.pop();
                regionMask.at<uchar>(current) = 1;
                uchar currentVal = img.at<uchar>(current);

                // Passo 3: Controllo dei pixel vicini
                for (int i = 0; i < 8; i++) {
                    Point neighbor = current + neighbors[i];

                    if (neighbor.x < 0 || neighbor.x >= img.cols || neighbor.y < 0 || neighbor.y >= img.rows)

                        // Salta pixel fuori immagine
                        continue;

                    if (labels.at<uchar>(neighbor) || regionMask.at<uchar>(neighbor))

                        // Salta pixel già etichettati o già nella regione
                        continue;

                    uchar neighborVal = img.at<uchar>(neighbor);
                    if (abs(int(currentVal) - int(neighborVal)) < simTH) {

                        // Aggiunge alla regione
                        regionMask.at<uchar>(neighbor) = 1;
                        points.push(neighbor);
                    }
                }
            }

            // Passo 4: Verifica della dimensione della regione
            int regionArea = int(sum(regionMask)[0]);
            if (regionArea > minArea) {

                // Assegna etichetta unica
                labels += regionMask * currentLabel;

                // Limite di etichette
                if (currentLabel++ > maxLabels)
                    return labels;

            } else

                // Regione considerata rumore
                labels += regionMask * 255;
        }

    return labels;
}
