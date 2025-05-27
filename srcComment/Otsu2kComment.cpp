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
 * 3. Calcola le probabilità e medie cumulative
 * 4. Calcola la media globale
 * 5. Trova le due soglie ottimali massimizzando la varianza tra classi
 * 6. Applica la sogliatura con i due valori trovati
 *
 * @param input Immagine in input (scala di grigi)
 * @return Immagine con pixel separati in sfondo (0), medio-piano (127) e primo piano (255)
 */
Mat otsu2k(const Mat &input) {

    // Passo 1: Applicazione dello sfocamento gaussiano per ridurre il rumore
    Mat img;
    GaussianBlur(input, img, Size(5, 5), 0.5);

    // Passo 2: Calcolo dell'istogramma normalizzato dell'immagine
    vector<double> histogram(256, 0.0);
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            histogram.at(img.at<uchar>(x, y))++;    // Incrementa il bin corrispondente al valore del pixel

    // Normalizzazione dell'istogramma
    int totalPixels = img.rows * img.cols;
    for (double &bin: histogram)
        bin /= totalPixels;     // Converti in probabilità dividendo per il numero totale di pixel

    // Passo 3: Calcolo delle probabilità e medie cumulative
    vector<double> cumProb(256, 0.0), cumMean(256, 0.0);
    cumProb[0] = histogram[0];
    cumMean[0] = 0.0;
    for (int i = 1; i < 256; ++i) {
        cumProb[i] = cumProb[i - 1] + histogram[i];
        cumMean[i] = cumMean[i - 1] + i * histogram[i];
    }

    // Passo 4: Calcolo della media globale
    double globalMean = cumMean[255];

    // Passo 5: Ricerca delle soglie ottimali k1 e k2
    double maxVariance = 0.0;
    int bestK1 = 0, bestK2 = 0;

    for (int k1 = 1; k1 < 254; ++k1)
        for (int k2 = k1 + 1; k2 < 255; ++k2) {

            // Calcolo delle probabilità delle tre classi
            double w0 = cumProb[k1];
            double w1 = cumProb[k2] - cumProb[k1];
            double w2 = 1.0 - cumProb[k2];

            // Calcolo delle medie delle tre classi
            double m0 = w0 > 0 ? cumMean[k1] / w0 : 0.0;
            double m1 = w1 > 0 ? (cumMean[k2] - cumMean[k1]) / w1 : 0;
            double m2 = w2 > 0 ? (cumMean[255] - cumMean[k2]) / w2 : 0;

            // Calcolo della varianza tra classi (criterio di Otsu)
            double betweenClassVariance = w0 * (m0 - globalMean) * (m0 - globalMean) +
                                          w1 * (m1 - globalMean) * (m1 - globalMean) +
                                          w2 * (m2 - globalMean) * (m2 - globalMean);

            // Aggiornamento delle soglie ottimali se si trova una varianza maggiore
            if (betweenClassVariance > maxVariance) {
                maxVariance = betweenClassVariance;
                bestK1 = k1;
                bestK2 = k2;
            }
        }

    // Passo 6: Sogliatura dell'immagine usando le due soglie ottimali
    Mat out = Mat::zeros(img.size(), CV_8U);
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++) {
            uchar pixel = img.at<uchar>(x, y);
            if (pixel >= bestK2)
                out.at<uchar>(x, y) = 255;  // Primo piano (bianco)
            else if (pixel >= bestK1)
                out.at<uchar>(x, y) = 127;  // Medio-piano (grigio)
            // Sfondo rimane nero (0)
        }

    return out;
}
