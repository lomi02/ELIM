#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica l'algoritmo di sogliatura a due livelli di Otsu (Otsu2k) a un'immagine in input.
 *
 * L'algoritmo divide l'immagine in tre regioni (sfondo, medio-piano, primo piano)
 * trovando automaticamente due soglie ottimali che massimizzano la varianza tra classi.
 *
 * Passaggi principali:
 * 1. Pre-processing con filtro Gaussiano
 * 2. Calcolo istogramma normalizzato
 * 3. Calcolo probabilità e medie cumulative
 * 4. Ricerca soglie ottimali k1 e k2
 * 5. Applicazione delle soglie all'immagine
 *
 * @param input Immagine in scala di grigi (CV_8UC1)
 * @return Immagine segmentata (0=sfondo, 127=medio-piano, 255=primo piano)
 */
Mat otsu2k(Mat &input) {

    // Passo 1: Pre-processing - Riduzione rumore con filtro Gaussiano
    Mat img = input.clone();
    GaussianBlur(img, img, Size(3, 3), 1);

    // Passo 2: Calcolo istogramma normalizzato
    vector hist(256, 0.0);
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            hist[img.at<uchar>(x, y)]++;  // Conteggio frequenze dei livelli di grigio

    // Normalizzazione dell'istogramma (conversione in probabilità)
    double totalPixels = img.rows * img.cols;
    for (double &val : hist)
        val /= totalPixels;

    // Passo 3: Calcolo probabilità cumulative (cumProb) e medie cumulative (cumMean)
    vector<double> cumProb(256, 0.0), cumMean(256, 0.0);
    cumProb[0] = hist[0];
    cumMean[0] = 0.0;
    for (int i = 1; i < 256; ++i) {
        cumProb[i] = cumProb[i - 1] + hist[i];      // Probabilità cumulativa
        cumMean[i] = cumMean[i - 1] + i * hist[i];  // Media cumulativa ponderata
    }

    // Passo 4: Calcolo media globale e inizializzazione variabili per la ricerca
    double globalMean = cumMean[255];   // Media globale dell'immagine
    double maxVar = 0.0;                // Massima varianza tra classi trovata
    vector th(2, 0);                    // Soglie ottimali (k1, k2)

    // Ricerca esaustiva delle soglie ottimali k1 e k2
    for (int k1 = 1; k1 < 254; ++k1)
        for (int k2 = k1 + 1; k2 < 255; ++k2) {
            // Calcolo probabilità delle tre classi:
            // w0: Pixels <= k1 (sfondo)
            // w1: k1 < Pixels <= k2 (medio-piano)
            // w2: Pixels > k2 (primo piano)
            double w0 = cumProb[k1];
            double w1 = cumProb[k2] - cumProb[k1];
            double w2 = 1.0 - cumProb[k2];

            // Calcolo medie delle classi (evitando divisioni per zero)
            double m0 = w0 > 0 ? cumMean[k1] / w0 : 0.0;
            double m1 = w1 > 0 ? (cumMean[k2] - cumMean[k1]) / w1 : 0.0;
            double m2 = w2 > 0 ? (cumMean[255] - cumMean[k2]) / w2 : 0.0;

            // Calcolo varianza inter-classe (criterio di Otsu)
            double var = w0 * (m0 - globalMean) * (m0 - globalMean) +
                         w1 * (m1 - globalMean) * (m1 - globalMean) +
                         w2 * (m2 - globalMean) * (m2 - globalMean);

            // Aggiornamento soglie se si trova una varianza maggiore
            if (var > maxVar) {
                maxVar = var;
                th[0] = k1;
                th[1] = k2;
            }
        }

    // Passo 5: Applicazione delle soglie all'immagine
    Mat out = Mat::zeros(img.size(), CV_8U);
    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            uchar pixel = img.at<uchar>(x, y);
            if (pixel >= th[1])
                out.at<uchar>(x, y) = 255;  // Primo piano (bianco)
            else if (pixel >= th[0])
                out.at<uchar>(x, y) = 127;  // Medio-piano (grigio)
            // Sfondo rimane 0 (nero)
        }
    }

    return out;
}
