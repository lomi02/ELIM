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
 * @param input      Immagine in input (scala di grigi)
 * @param houghTH    Soglia per il rilevamento nello spazio di Hough
 * @param radiusMin  Raggio minimo dei cerchi da rilevare
 * @param radiusMax  Raggio massimo dei cerchi da rilevare
 * @param cannyTHL   Soglia inferiore per il rilevamento bordi di Canny
 * @param cannyTHH   Soglia superiore per il rilevamento bordi di Canny
 * @param blurSize   Dimensione del kernel gaussiano per lo smoothing (default 3)
 * @param blurSigma  Deviazione standard per lo sfocamento gaussiano (default 0.5)
 *
 * @return Immagine con i cerchi rilevati disegnati
 */
Mat hough_circles(Mat &input, int houghTH, int radiusMin, int radiusMax, int cannyTHL, int cannyTHH, int blurSize, float blurSigma) {
    Mat img = input.clone();

    // Passo 1: Applicazione dello sfocamento gaussiano per ridurre il rumore
    // Lo smoothing è importante per migliorare il successivo rilevamento dei bordi
    GaussianBlur(img, img, Size(blurSize, blurSize), blurSigma);

    // Passo 2: Rilevamento dei bordi con l'algoritmo di Canny
    // Canny trova i bordi significativi usando le due soglie (bassa e alta)
    Canny(img, img, cannyTHL, cannyTHH);

    // Passo 3: Calcolo della Trasformata di Hough per rilevare i cerchi
    int radiusOffset = radiusMax - radiusMin + 1;
    int sizes[] = {img.cols, img.rows, radiusOffset};   // Dimensioni dell'accumulatore 3D
    auto votes = Mat(3, sizes, CV_8U, Scalar(0));       // Matrice dei voti (accumulatore)

    // Per ogni pixel dell'immagine...
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)

            // Se è un pixel di bordo (valore 255)...
            if (img.at<uchar>(Point(x, y)) == 255)

                // Per ogni possibile raggio nel range specificato...
                for (int radius = radiusMin; radius < radiusMax; ++radius)

                    // Per ogni possibile angolo theta (0-360 gradi)...
                    for (int thetaDegrees = 0; thetaDegrees < 360; ++thetaDegrees) {
                        double thetaRadians = thetaDegrees * CV_PI / 180;   // Conversione in radianti

                        // Calcola le coordinate del centro (alpha, beta) del cerchio potenziale
                        int alpha = cvRound(x - radius * cos(thetaRadians));
                        int beta = cvRound(y - radius * sin(thetaRadians));

                        // Verifica che il centro sia dentro i bordi dell'immagine
                        if (alpha >= 0 and alpha < img.cols and beta >= 0 and beta < img.rows)

                            // Incrementa il voto per questo centro e raggio
                            votes.at<uchar>(beta, alpha, radius - radiusMin)++;
                    }

    // Passo 4: Disegna i cerchi rilevati sull'immagine originale
    Mat out = input.clone();

    // Itera su tutti i possibili raggi...
    for (int radius = radiusMin; radius < radiusMax; ++radius)

        // Itera su tutte le possibili coordinate del centro...
        for (int alpha = 0; alpha < img.cols; ++alpha)
            for (int beta = 0; beta < img.rows; ++beta)

                // Se i voti superano la soglia, abbiamo trovato un cerchio
                    if (votes.at<uchar>(beta, alpha, radius - radiusMin) > houghTH)

                    // Disegna il cerchio con il raggio e centro rilevati
                    circle(out, Point(alpha, beta), radius, Scalar(0), 2, 8);

    return out;
}
