#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Implementa l'algoritmo di region growing per la segmentazione di immagini.
 *
 * Parte da un punto seed e propaga la regione considerando i pixel adiacenti
 * la cui intensità è simile (in base a una soglia). Usa una strategia BFS.
 *
 * @param input     Immagine di input in scala di grigi (8-bit)
 * @param similTH   Soglia di similarità per accettare un pixel nella regione
 * @param seed      Punto di partenza da cui inizia la crescita della regione
 * @return          Immagine binaria (255 = pixel nella regione, 0 = sfondo)
 */
Mat region_growing(Mat &input, int similTH, Point seed) {

    // Crea una matrice per tenere traccia dei pixel già visitati
    Mat img = Mat::zeros(input.size(), CV_8U);

    // Coda per la BFS (Breadth-First Search)
    queue<Point> pixelQueue;
    pixelQueue.push(seed);          // Aggiunge il seed alla coda
    img.at<uchar>(seed) = 1;    // Marca il seed come visitato

    // Definizione dei 8 vicini (connessione 8-way)
    Point neighbors[8] = {
        Point(-1, -1), Point(-1, 0), Point(-1, 1),
        Point(0, -1),                Point(0, 1),
        Point(1, -1), Point(1, 0), Point(1, 1)
    };

    // Immagine di output: inizialmente tutta nera (0)
    Mat out = Mat::zeros(input.size(), CV_8U);

    // Algoritmo principale - esplora i pixel finché la coda non è vuota
    while (!pixelQueue.empty()) {

        // Estrae il pixel corrente dalla coda
        Point currentPx = pixelQueue.front();
        pixelQueue.pop();

        // Marca il pixel corrente come parte della regione (255 = bianco)
        out.at<uchar>(currentPx) = 255;

        // Scorre tutti i vicini
        for (Point &offset : neighbors) {
            Point neighPx = currentPx + offset;

            // Verifica le condizioni per includere il vicino:
            // 1. Deve essere dentro i bordi dell'immagine
            // 2. Non deve essere stato visitato
            // 3. La differenza di intensità deve essere <= soglia
            if (neighPx.x >= 0 && neighPx.x < input.cols &&
                neighPx.y >= 0 && neighPx.y < input.rows &&
                img.at<uchar>(neighPx) == 0 &&
                abs(input.at<uchar>(currentPx) - input.at<uchar>(neighPx)) <= similTH) {

                img.at<uchar>(neighPx) = 1;     // Marca il vicino come visitato
                pixelQueue.push(neighPx);           // Aggiunge alla coda per processarlo
            }
        }
    }

    return out;  // Restituisce la regione segmentata come immagine binaria
}
