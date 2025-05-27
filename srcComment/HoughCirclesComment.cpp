#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

/**
 * Applica l'algoritmo di rilevamento cerchi di Hough a un'immagine in input.
 *
 * Questa funzione esegue i seguenti passaggi:
 * 1. Applica uno sfocamento gaussiano per ridurre il rumore
 * 2. Esegue il rilevamento dei bordi con Canny
 * 3. Calcola la Trasformata di Hough per individuare i cerchi
 * 4. Disegna i cerchi rilevati sull'immagine originale
 *
 * @param input    Immagine in input (scala di grigi)
 * @param houghTH  Soglia per il rilevamento nello spazio di Hough
 * @param Rmin     Raggio minimo dei cerchi da rilevare
 * @param Rmax     Raggio massimo dei cerchi da rilevare
 *
 * @return Immagine con i cerchi rilevati disegnati
 */
Mat hough_circles(Mat &input, int houghTH, int Rmin, int Rmax) {
    Mat img = input.clone();

    // Passo 1: Applicazione dello sfocamento gaussiano per ridurre il rumore
    // Lo smoothing è importante per migliorare il successivo rilevamento dei bordi
    GaussianBlur(img, img, Size(5, 5), 0.5, 0.5);

    // Passo 2: Rilevamento dei bordi con l'algoritmo di Canny
    // Canny trova i bordi significativi usando soglie fisse (100 e 250)
    Canny(img, img, 100, 250);

    // Passo 3: Calcolo della Trasformata di Hough per rilevare i cerchi
    // Inizializza la matrice dei voti (accumulatore) come vettore 3D
    vector votes(img.rows, vector(img.cols, vector(Rmax - Rmin + 1, 0)));

    // Per ogni pixel dell'immagine...
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)

            // Se è un pixel di bordo (valore 255)...
            if (img.at<uchar>(x, y) == 255)

                // Per ogni possibile raggio nel range specificato...
                for (int r = Rmin; r < Rmax; r++)

                    // Per ogni possibile angolo theta (0-360 gradi)...
                    for (int theta = 0; theta < 360; theta++) {

                        // Calcola le coordinate del centro (a, b) del cerchio potenziale
                        int a = y - r * cos(theta * CV_PI / 180);
                        int b = x - r * sin(theta * CV_PI / 180);

                        // Verifica che il centro sia dentro i bordi dell'immagine
                        if (a >= 0 && a < img.rows && b >= 0 && b < img.cols)

                            // Incrementa il voto per questo centro e raggio
                            votes[a][b][r - Rmin]++;
                    }

    // Passo 4: Disegna i cerchi rilevati sull'immagine originale
    Mat out = input.clone();

    // Itera su tutte le possibili coordinate del centro...
    for (int a = 0; a < img.rows; a++)
        for (int b = 0; b < img.cols; b++)

            // Itera su tutti i possibili raggi...
            for (int r = Rmin; r < Rmax; r++)

                // Se i voti superano la soglia, abbiamo trovato un cerchio
                if (votes[a][b][r - Rmin] > houghTH)

                    // Disegna il cerchio con il raggio e centro rilevati
                    circle(out, Point(a, b), r, Scalar(0));

    return out;
}
