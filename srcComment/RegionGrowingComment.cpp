#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Controlla se un punto è all'interno dei confini dell'immagine.
 *
 * @param img   L'immagine di riferimento per le dimensioni
 * @param neigh Il punto da verificare
 * @return      True se il punto è dentro i limiti, false altrimenti
 */
bool inRange(const Mat &img, Point neigh) {
    return neigh.x >= 0 && neigh.x < img.cols &&    // Controllo coordinata x
           neigh.y >= 0 && neigh.y < img.rows;      // Controllo coordinata y
}

/**
 * Verifica la similarità di intensità tra due pixel.
 *
 * @param img       Immagine di riferimento
 * @param p1        Primo pixel da confrontare
 * @param p2        Secondo pixel da confrontare
 * @param similTH   Soglia di differenza di intensità consentita
 * @return          True se la differenza è entro la soglia, false altrimenti
 */
bool isSimilar(const Mat &img, Point p1, Point p2, int similTH) {

    // Calcola la differenza assoluta tra le intensità
    int intensityDelta = abs(img.at<uchar>(p1) - img.at<uchar>(p2));
    return intensityDelta <= similTH;   // Confronto con la soglia
}

/**
 * Implementa l'algoritmo di region growing per la segmentazione di immagini.
 *
 * L'algoritmo parte da un pixel seed e cresce la regione includendo pixel vicini
 * con intensità simili, usando un approccio breadth-first search.
 *
 * @param input     Immagine di input in scala di grigi (8-bit)
 * @param similTH   Soglia di similarità per l'inclusione nella regione
 * @param seed      Punto di partenza (default: angolo in alto a sinistra)
 * @return          Immagine binaria con la regione segmentata (255) e lo sfondo (0)
 */
Mat region_growing(const Mat &input, int similTH, Point seed = Point(0, 0)) {
    Mat img = input.clone();
    Mat out = Mat::zeros(img.size(), CV_8U);

    // Matrice per tracciare i pixel già visitati
    Mat visited = Mat::zeros(img.size(), CV_8U);

    // Coda per l'implementazione BFS (Breadth-First Search)
    queue<Point> pixelQueue;
    pixelQueue.push(seed);          // Inizia dal pixel seed
    visited.at<uchar>(seed) = 1;    // Marca il seed come visitato

    // Definizione degli 8 vicini (connessione 8-way)
    Point neighbors[] = {
        Point(0, -1), Point(-1, 0),
        Point(1, 0),  Point(0, 1)
    };

    // Algoritmo principale - continua finché ci sono pixel da processare
    while (!pixelQueue.empty()) {

        // Prende il prossimo pixel dalla coda
        Point currentPx = pixelQueue.front();
        pixelQueue.pop();

        // Marca il pixel come parte della regione (255 = bianco)
        out.at<uchar>(currentPx) = 255;

        // Esplora tutti gli 8 vicini
        for (const Point &offset : neighbors) {
            Point neighPx = currentPx + offset;

            // Condizioni per includere il vicino:
            // 1. Deve essere dentro l'immagine
            // 2. Non deve essere già stato visitato
            // 3. Deve avere intensità simile al pixel corrente
            if (inRange(img, neighPx) &&
                visited.at<uchar>(neighPx) == 0 &&
                isSimilar(img, currentPx, neighPx, similTH)) {

                visited.at<uchar>(neighPx) = 1; // Marca come visitato
                pixelQueue.push(neighPx);       // Aggiunge alla coda
            }
        }
    }

    return out;  // Restituisce la mappa binaria della regione
}