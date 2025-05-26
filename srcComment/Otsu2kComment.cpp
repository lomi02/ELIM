#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica l'algoritmo di sogliatura a due livelli di Otsu (Otsu2k) a un'immagine in input.
 *
 * L'algoritmo estende il metodo di Otsu classico dividendo l'immagine in tre regioni:
 * sfondo, medio-piano e primo piano, trovando due valori di soglia ottimali.
 *
 * La funzione esegue questi passaggi:
 * 1. Applica uno sfocatura gaussiana per ridurre il rumore
 * 2. Calcola l'istogramma normalizzato dell'immagine
 * 3. Calcola la media cumulativa globale
 * 4. Calcola iterativamente le probabilità e medie cumulative per diverse combinazioni
 *    di soglia, massimizzando la varianza tra le classi
 * 5. Sogliatura l'immagine sfocata usando i due valori di soglia ottimali
 *
 * @param input     Immagine in input (scala di grigi)
 * @param blurSize  Dimensione del kernel gaussiano per lo smoothing (default 3)
 * @param blurSigma Deviazione standard per lo sfocamento gaussiano (default 0.5)
 *
 * @return Immagine binaria con pixel separati in sfondo, oggetto e primo piano
 */
Mat otsu2k(const Mat &input, int blurSize, float blurSigma) {

    // Passo 1: Applicazione dello sfocamento gaussiano per ridurre il rumore
    Mat img = input.clone();
    GaussianBlur(img, img, Size(blurSize, blurSize), blurSigma);

    // Passo 2: Calcolo dell'istogramma normalizzato dell'immagine
    vector histogram(256, 0.0);
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            histogram[img.at<uchar>(i, j)]++;

    // Normalizzazione dell'istogramma
    double totalPixels = img.rows * img.cols;
    for (auto &histValue: histogram)
        histValue /= totalPixels;

    // Passo 3: Calcolo della media cumulativa globale (mg)
    double globalMean = 0.0;
    for (int i = 0; i < 256; i++)
        globalMean += i * histogram[i];

    // Inizializzazione variabili per il calcolo delle soglie ottimali
    double maxVar = 0.0;        // Massima varianza tra classi trovata
    vector optimalTH = {0, 0};  // Soglie ottimali (k1, k2)

    // Variabili temporanee per il calcolo
    double w0, w1, w2, m0, m1, m2, variance;

    // Passo 4: Ricerca iterativa delle soglie ottimali
    for (int k1 = 1; k1 < 256 - 2; k1++) {
        w0 = m0 = 0.0;

        // Calcola probabilità e media per la prima classe (sfondo)
        for (int i = 0; i <= k1; i++) {
            w0 += histogram[i];
            m0 += i * histogram[i];
        }

        for (int k2 = k1 + 1; k2 < 256 - 1; k2++) {
            w1 = m1 = 0.0;

            // Calcola probabilità e media per la seconda classe (medio-piano)
            for (int i = k1 + 1; i <= k2; i++) {
                w1 += histogram[i];
                m1 += i * histogram[i];
            }

            // Calcola probabilità e media per la terza classe (primo piano)
            w2 = 1.0 - w0 - w1;
            m2 = globalMean - m0 - m1;

            // Calcolo della varianza tra classi (criterio di Otsu)
            variance = w0 * (m0 / w0 - globalMean) * (m0 / w0 - globalMean) +
                       w1 * (m1 / w1 - globalMean) * (m1 / w1 - globalMean) +
                       w2 * (m2 / w2 - globalMean) * (m2 / w2 - globalMean);

            // Aggiornamento delle soglie ottimali se si trova una varianza maggiore
            if (variance > maxVar) {
                maxVar = variance;
                optimalTH = {k1, k2};
            }
        }
    }

    // Passo 5: Sogliatura dell'immagine usando le due soglie ottimali
    Mat out = Mat::zeros(img.size(), CV_8U);
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++) {
            uchar val = img.at<uchar>(i, j);

            // Primo piano (bianco)
            if (val >= optimalTH[1])
                out.at<uchar>(i, j) = 255;

            // Medio-piano (grigio)
            else if (val >= optimalTH[0])
                out.at<uchar>(i, j) = 127;

            // Sfondo rimane nero (0)
        }

    return out;
}
