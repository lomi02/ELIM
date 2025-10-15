#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica l'algoritmo di Otsu per la sogliatura automatica di un'immagine in scala di grigi.
 *
 * L’algoritmo di Otsu trova automaticamente una soglia ottimale che separa i pixel
 * in due classi (sfondo e oggetto) massimizzando la varianza inter-classe.
 *
 * La procedura segue questi passaggi:
 * 1. Applica una sfocatura gaussiana per ridurre il rumore
 * 2. Calcola l’istogramma dell’immagine
 * 3. Normalizza l’istogramma per ottenere una distribuzione di probabilità
 * 4. Calcola la media globale dei livelli di grigio
 * 5. Scansiona tutti i possibili valori di soglia per trovare quello che massimizza la varianza inter-classe
 * 6. Applica la soglia ottimale per binarizzare l’immagine
 *
 * @param input  Immagine di input in scala di grigi
 * @return       Immagine binaria (0 e 255) ottenuta con la soglia di Otsu
 */
Mat otsu(Mat &input) {
    Mat img = input.clone();

    // Passo 1: Applicazione di un filtro gaussiano
    // Riduce il rumore e migliora la stabilità della sogliatura
    GaussianBlur(img, img, Size(3, 3), 0.5, 0.5);

    // Passo 2: Calcolo dell'istogramma
    // Conta il numero di pixel per ogni livello di intensità (0–255)
    vector hist(256, 0.0);
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            hist[img.at<uchar>(x, y)]++;

    // Passo 3: Normalizzazione dell'istogramma
    // Converte le frequenze in probabilità (somma totale = 1)
    double totalPixels = img.rows * img.cols;
    for (double &val : hist)
        val /= totalPixels;

    // Passo 4: Calcolo della media globale dell’immagine
    // μT = Σ(i * p(i)) per i = 0..255
    double globalMean = 0.0;
    for (int i = 0; i < 256; i++)
        globalMean += i * hist[i];

    // Passo 5: Ricerca della soglia ottimale che massimizza la varianza inter-classe
    // Si scansionano tutte le possibili soglie k
    double prob = 0.0;       // Probabilità cumulativa ω(k)
    double cumMean = 0.0;    // Media cumulativa μ(k)
    double maxVar = 0.0;     // Massima varianza inter-classe trovata
    int bestThreshold = 0;   // Soglia ottimale

    for (int k = 0; k < 256; k++) {
        prob += hist[k];           // Aggiorna la probabilità cumulativa
        cumMean += k * hist[k];    // Aggiorna la media cumulativa

        // Calcolo della varianza inter-classe:
        // σ_B² = [μT * ω(k) - μ(k)]² / [ω(k) * (1 - ω(k))]
        // (formula derivata da Otsu)
        if (prob > 0.0 && prob < 1.0) {  // Evita divisione per zero
            double betweenVar = pow(globalMean * prob - cumMean, 2) / (prob * (1.0 - prob));

            // Aggiornamento della soglia ottimale
            if (betweenVar > maxVar) {
                maxVar = betweenVar;
                bestThreshold = k;
            }
        }
    }

    // Passo 6: Applicazione della soglia ottimale all'immagine
    // Tutti i pixel >= soglia vengono impostati a 255, gli altri a 0
    Mat out;
    threshold(img, out, bestThreshold, 255, THRESH_BINARY);

    // Restituisce l’immagine binaria finale
    return out;
}
