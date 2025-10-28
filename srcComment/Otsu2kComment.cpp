#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica una versione estesa del metodo di Otsu per segmentare un'immagine in tre classi.
 *
 * Questa funzione esegue i seguenti passaggi:
 * 1. Riduce il rumore dell'immagine tramite filtro gaussiano
 * 2. Calcola l'istogramma normalizzato dei livelli di grigio
 * 3. Calcola la media globale dei pixel
 * 4. Determina le due soglie ottimali massimizzando la varianza inter-classi (3 classi)
 * 5. Assegna i pixel a tre livelli di intensità: 0, 127, 255
 *
 * @param input  Immagine in input (scala di grigi)
 *
 * @return Immagine segmentata in tre classi usando le due soglie di Otsu
 */
Mat otsu2k(Mat &input) {
    Mat img = input.clone();

    // Passo 1: Riduzione del rumore tramite filtro gaussiano 3x3
    GaussianBlur(img, img, Size(3, 3), 0.5, 0.5);

    // Passo 2: Calcolo dell'istogramma normalizzato dei pixel
    vector<double> hist(256, 0.0);
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            hist[img.at<uchar>(x, y)]++; // Incrementa il conteggio del livello di grigio

    double tot = img.rows * img.cols;
    for (int i = 0; i < 256; i++)
        hist[i] /= tot; // Normalizzazione in [0,1]

    // Passo 3: Calcolo della media globale dei pixel
    double gMean = 0.0;
    for (int i = 0; i < 256; i++)
        gMean += i * hist[i];

    // Passo 4: Ricerca delle due soglie ottimali massimizzando la varianza inter-classi
    double maxVar = 0.0;
    int bestTH1 = 0, bestTH2 = 0;

    // Ciclo su tutte le possibili coppie di soglie t1 < t2
    for (int t1 = 0; t1 < 255; t1++)
        for (int t2 = t1 + 1; t2 < 256; t2++) {
            double w0 = 0, w1 = 0, w2 = 0;        // Pesature delle tre classi
            double sum0 = 0, sum1 = 0, sum2 = 0;  // Somma dei valori dei pixel per ciascuna classe

            // Classe 0: pixel <= t1
            for (int i = 0; i <= t1; i++) {
                w0 += hist[i];
                sum0 += i * hist[i];
            }

            // Classe 1: t1 < pixel <= t2
            for (int i = t1 + 1; i <= t2; i++) {
                w1 += hist[i];
                sum1 += i * hist[i];
            }

            // Classe 2: pixel > t2
            for (int i = t2 + 1; i < 256; i++) {
                w2 += hist[i];
                sum2 += i * hist[i];
            }

            // Calcolo della varianza inter-classi solo se tutte le classi hanno peso > 0
            if (w0 > 0 && w1 > 0 && w2 > 0) {
                double mean0 = sum0 / w0;
                double mean1 = sum1 / w1;
                double mean2 = sum2 / w2;

                double var = w0 * pow(mean0 - gMean, 2) +
                             w1 * pow(mean1 - gMean, 2) +
                             w2 * pow(mean2 - gMean, 2);

                // Aggiornamento delle soglie ottimali
                if (var > maxVar) {
                    maxVar = var;
                    bestTH1 = t1;
                    bestTH2 = t2;
                }
            }
        }

    // Passo 5: Assegnazione dei pixel alle tre classi
    Mat out = img.clone();
    for (int x = 0; x < out.rows; x++)
        for (int y = 0; y < out.cols; y++) {
            uchar val = out.at<uchar>(x, y);
            if (val <= bestTH1)
                out.at<uchar>(x, y) = 0;      // Classe 0
            else if (val <= bestTH2)
                out.at<uchar>(x, y) = 127;    // Classe 1
            else
                out.at<uchar>(x, y) = 255;    // Classe 2
        }

    return out;
}
