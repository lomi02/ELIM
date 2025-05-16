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
 * 1. Calcola l'istogramma normalizzato dell'immagine
 * 2. Calcola la media cumulativa globale
 * 3. Calcola iterativamente le probabilità e medie cumulative per diverse combinazioni
 *    di soglia, massimizzando la varianza tra le classi
 * 4. Applica uno sfocatura gaussiana per ridurre il rumore
 * 5. Sogliatura l'immagine sfocata usando i due valori di soglia ottimali
 *
 * @param input     Immagine in input (scala di grigi)
 * @param blurSize  Dimensione del kernel gaussiano per lo smoothing (default 3)
 * @param blurSigma Deviazione standard per lo sfocamento gaussiano (default 0.5)
 *
 * @return Immagine binaria con pixel separati in sfondo, oggetto e primo piano
 */
Mat otsu2k(Mat &input, int blurSize = 3, float blurSigma = 0.5) {
    Mat img = input.clone();

    // Passo 1: Calcolo dell'istogramma normalizzato dell'immagine
    vector histogram(256, 0.0);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            histogram.at(img.at<uchar>(Point(x, y)))++;     // Incrementa il bin corrispondente al valore del pixel

    // Normalizzazione dell'istogramma
    int numberOfPixels = img.rows * img.cols;
    for (double & bin : histogram)
        bin /= numberOfPixels;      // Converti in probabilità dividendo per il numero totale di pixel

    // Passo 2: Calcolo della media cumulativa globale (mg)
    double globalCumulativeMean = 0.0;
    for (int i = 0; i < histogram.size(); ++i)
        globalCumulativeMean += i * histogram.at(i);    // Somma ponderata dei valori di intensità

    // Inizializzazione variabili per il calcolo delle soglie ottimali
    vector probabilities(3, 0.0);       // Probabilità delle tre classi (P0, P1, P2)
    vector cumulativeMeans(3, 0.0);     // Medie cumulative (m0, m1, m2)
    double maxVariance = 0.0;           // Massima varianza tra classi trovata
    vector optimalTH(2, 0);             // Soglie ottimali (k1, k2)

    // Passo 3: Ricerca iterativa delle soglie ottimali
    for (int k1 = 0; k1 < histogram.size() - 2; ++k1) {

        // Aggiorna probabilità e media cumulativa per la prima classe (sfondo)
        probabilities.at(0) += histogram.at(k1);
        cumulativeMeans.at(0) += k1 * histogram.at(k1);

        for (int k2 = k1 + 1; k2 < histogram.size() - 1; ++k2) {

            // Aggiorna probabilità e media cumulativa per la seconda classe (medio-piano)
            probabilities.at(1) += histogram.at(k2);
            cumulativeMeans.at(1) += k2 * histogram.at(k2);

            for (int k = k2 + 1; k < histogram.size(); ++k) {

                // Aggiorna probabilità e media cumulativa per la terza classe (primo piano)
                probabilities.at(2) += histogram.at(k);
                cumulativeMeans.at(2) += k * histogram.at(k);

                // Calcolo della varianza tra classi (criterio di Otsu)
                double betweenClassesVariance = 0.0;
                for (int i = 0; i < 3; ++i) {
                    if (probabilities.at(i) > 0) {
                        double currentCumulativeMean = cumulativeMeans.at(i) / probabilities.at(i);
                        betweenClassesVariance += probabilities.at(i) * pow(currentCumulativeMean - globalCumulativeMean, 2);
                    }
                }

                // Aggiornamento delle soglie ottimali se si trova una varianza maggiore
                if (betweenClassesVariance > maxVariance) {
                    maxVariance = betweenClassesVariance;
                    optimalTH.at(0) = k1;
                    optimalTH.at(1) = k2;
                }
            }
            // Reset per la terza classe per la prossima iterazione
            probabilities.at(2) = 0.0;
            cumulativeMeans.at(2) = 0.0;
        }
        // Reset per la seconda classe per la prossima iterazione
        probabilities.at(1) = 0.0;
        cumulativeMeans.at(1) = 0.0;
    }

    // Passo 4: Applicazione dello sfocamento gaussiano per ridurre il rumore
    GaussianBlur(img, img, Size(blurSize, blurSize), blurSigma, blurSigma);

    // Passo 5: Sogliatura dell'immagine usando le due soglie ottimali
    Mat thresholdedImg = Mat::zeros(img.rows, img.cols, CV_8U);
    for (int x = 0; x < img.rows; ++x) {
        for (int y = 0; y < img.cols; ++y) {
            uchar pixel = img.at<uchar>(x, y);
            if (pixel >= optimalTH.at(1))
                thresholdedImg.at<uchar>(x, y) = 255;   // Primo piano (bianco)
            else if (pixel >= optimalTH.at(0))
                thresholdedImg.at<uchar>(x, y) = 128;   // Medio-piano (grigio)
            // Sfondo rimane nero (0)
        }
    }
    return thresholdedImg;
}
