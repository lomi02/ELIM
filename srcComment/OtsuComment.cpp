#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica il metodo di Otsu per la binarizzazione automatica di un'immagine in scala di grigi.
 *
 * Questa funzione esegue i seguenti passaggi:
 * 1. Applica un filtro gaussiano 3x3 per ridurre il rumore dell'immagine.
 * 2. Calcola l'istogramma normalizzato dei livelli di intensità.
 * 3. Calcola la media globale dei livelli di grigio.
 * 4. Determina la soglia ottimale massimizzando la varianza inter-classi tramite scansione dei possibili livelli.
 * 5. Applica la soglia ottenuta per generare l'immagine binaria.
 *
 * @param input Immagine in scala di grigi.
 * @return Immagine binaria ottenuta tramite il metodo di Otsu.
 */
Mat otsu(Mat &input) {
    Mat img = input.clone();

    // Passo 1: Riduzione del rumore con filtro gaussiano 3x3
    GaussianBlur(img, img, Size(3, 3), 0.5, 0.5);

    // Passo 2: Calcolo dell'istogramma normalizzato
    vector<double> hist(256, 0.0);
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            hist[img.at<uchar>(x, y)]++;

    double tot = img.rows * img.cols;
    for (int i = 0; i < 256; i++)
        hist[i] /= tot;

    // Passo 3: Media globale dei livelli di grigio
    double gMean = 0.0;
    for (int i = 0; i < 256; i++)
        gMean += i * hist[i];

    // Passo 4: Ricerca della soglia che massimizza la varianza inter-classi
    double w = 0.0;       // Probabilità cumulativa classe 0
    double cMean = 0.0;   // Media cumulativa classe 0
    double maxVar = 0.0;  // Varianza inter-classi massima
    int bestTH = 0;       // Soglia ottimale

    for (int i = 0; i < 256; i++) {
        w += hist[i];

        if (w > 0.0 && w < 1.0) {
            cMean += i * hist[i];

            // Formula della varianza inter-classi
            double var = pow(gMean * w - cMean, 2) / (w * (1.0 - w));
            if (var > maxVar) {
                maxVar = var;
                bestTH = i;
            }
        }
    }

    // Passo 5: Applicazione della soglia scelta
    Mat out;
    threshold(img, out, bestTH, 255, THRESH_BINARY);

    return out;
}
