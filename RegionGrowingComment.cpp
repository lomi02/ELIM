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
bool inRange(Mat &img, Point neigh) {
    return neigh.x >= 0 && neigh.x < img.cols &&   // Controllo coordinata x
           neigh.y >= 0 && neigh.y < img.rows;     // Controllo coordinata y
}

/**
 * Verifica la similarità di intensità tra due pixel.
 *
 * @param img       Immagine di riferimento
 * @param seed      Punto di partenza (pixel seed)
 * @param neigh     Pixel vicino da confrontare
 * @param similTH   Soglia di differenza di intensità consentita
 * @return          True se la differenza è entro la soglia, false altrimenti
 */
bool isSimilar(Mat &img, Point seed, Point neigh, int similTH) {

    // Estrae l'intensità dei due pixel
    int seedIntensity = img.at<uchar>(seed);
    int currIntensity = img.at<uchar>(neigh);

    // Calcola la differenza assoluta
    int intensityDelta = abs(seedIntensity - currIntensity);

    return intensityDelta <= similTH;  // Confronto con la soglia
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
Mat region_growing(Mat &input, int similTH, Point seed = Point(0, 0)) {
    Mat img = input.clone();

    // Inizializza l'immagine risultato (tutto zero = sfondo nero)
    Mat out = Mat::zeros(img.size(), CV_8U);

    // Coda per l'implementazione BFS (Breadth-First Search)
    queue<Point> pixelQueue;
    pixelQueue.push(seed);  // Inizia dal pixel seed

    // Algoritmo principale - continua finché ci sono pixel da processare
    while (!pixelQueue.empty()) {

        // Prende il prossimo pixel dalla coda
        Point currentPx = pixelQueue.front();
        pixelQueue.pop();

        // Processa solo pixel non ancora visitati (segno = 0)
        if (out.at<uchar>(currentPx) == 0) {

            // Marca il pixel come parte della regione (255 = bianco)
            out.at<uchar>(currentPx) = 255;

            // Definisce una finestra 3x3 attorno al pixel corrente
            Rect roi(currentPx.x - 1, currentPx.y - 1, 3, 3);

            // Scansione dei vicini nell'intorno 3x3
            for (int roi_x = roi.x; roi_x < roi.x + roi.height; ++roi_x)
                for (int roi_y = roi.y; roi_y < roi.y + roi.width; ++roi_y) {
                    Point neighPx(roi_x, roi_y);

                    // Condizioni per includere il vicino:
                    // 1. Deve essere dentro l'immagine
                    // 2. Deve avere intensità simile al seed
                    // 3. Non deve essere il pixel centrale stesso
                    if (inRange(img, neighPx) and isSimilar(img, seed, neighPx, similTH))

                        // Aggiunge il pixel valido alla coda per esplorazione
                        pixelQueue.push(neighPx);
                    }

        }
    }
    return out;  // Restituisce la mappa binaria della regione
}
