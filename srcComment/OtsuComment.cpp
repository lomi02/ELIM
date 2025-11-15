#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica il metodo di Otsu per la binarizzazione automatica di un'immagine in scala di grigi.
 *
 * Questa funzione esegue i seguenti passaggi:
 * 1. Riduce il rumore dell'immagine tramite filtro gaussiano
 * 2. Calcola l'istogramma normalizzato dei livelli di grigio
 * 3. Calcola la media globale dei pixel
 * 4. Determina la soglia ottimale massimizzando la varianza inter-classi
 * 5. Applica la soglia per ottenere un'immagine binaria
 *
 * @param input  Immagine in input (scala di grigi)
 *
 * @return Immagine binaria ottenuta con la soglia di Otsu
 */
Mat otsu(Mat &input) {
    Mat img = input.clone();

    // Passo 1: Riduzione del rumore tramite filtro gaussiano 3x3
    GaussianBlur(img, img, Size(3, 3), 0.5, 0.5);

    // Passo 2: Calcolo dell'istogramma normalizzato dei pixel
    vector<double> hist(256, 0.0);
    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            hist[img.at<uchar>(x, y)] += 1.0;   // Incrementa il conteggio del livello di grigio
        }
    }

    double tot = img.rows * img.cols; // Numero totale di pixel
    for (size_t i = 0; i < hist.size(); i++)
        hist[i] /= tot; // Normalizzazione in [0,1]

    // Passo 3: Calcolo della media globale dei pixel
    double gMean = 0.0;
    for (int i = 0; i < 256; i++)
        gMean += i * hist[i];

    // Passo 4: Ricerca della soglia ottimale massimizzando la varianza inter-classi
    double w = 0.0;       // Probabilità cumulativa della classe 0
    double cMean = 0.0;   // Media cumulativa della classe 0
    double maxVar = 0.0;  // Varianza inter-classi massima
    int bestTH = 0;       // Soglia ottimale

    for (int k = 0; k < 256; k++) {
        w += hist[k];               // Aggiorna peso classe 0

        // Evita divisione per zero
        if (w > 0.0 && w < 1.0) {
            cMean += k * hist[k];       // Aggiorna media cumulativa classe 0

            // Varianza inter-classi
            double var = pow(gMean * w - cMean, 2) / (w * (1.0 - w));
            if (var > maxVar) {
                maxVar = var;
                bestTH = k;         // Aggiorna soglia ottimale
            }
        }
    }

    // Passo 5: Applicazione della soglia ottimale
    Mat out;
    threshold(img, out, bestTH, 255, THRESH_BINARY);

    return out;
}