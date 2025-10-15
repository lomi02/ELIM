#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica una versione estesa dell'algoritmo di Otsu a due soglie (Otsu a 3 classi).
 *
 * Questa funzione suddivide l'immagine in tre classi d’intensità (ad esempio: sfondo, livello intermedio e oggetti chiari),
 * determinando due soglie ottimali t1 e t2 che massimizzano la varianza tra-classi.
 *
 * La procedura segue questi passaggi:
 * 1. Applica una sfocatura gaussiana per ridurre il rumore
 * 2. Calcola l'istogramma dei livelli di grigio
 * 3. Normalizza l'istogramma in modo che la somma sia 1
 * 4. Calcola la media globale dell'immagine
 * 5. Ricerca le due soglie (t1, t2) che massimizzano la varianza inter-classe
 * 6. Classifica i pixel in tre livelli in base alle soglie trovate
 *
 * @param input      Immagine in scala di grigi
 * @param blurSize   Dimensione del kernel gaussiano per la sfocatura (default 3)
 * @param blurSigma  Deviazione standard per la sfocatura gaussiana (default 0.5)
 *
 * @return Immagine segmentata con tre livelli di intensità: 0, 127, 255
 */
Mat otsu2k(Mat &input, int blurSize = 3, float blurSigma = 0.5) {
    Mat img = input.clone();

    // Passo 1: Applicazione del filtro gaussiano
    // Riduce il rumore e le variazioni locali per una sogliatura più stabile
    GaussianBlur(img, img, Size(blurSize, blurSize), blurSigma, blurSigma);

    // Passo 2: Calcolo dell'istogramma dei livelli di grigio
    // L'istogramma contiene la frequenza di ogni livello (0–255)
    vector hist(256, 0.0);
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            hist[img.at<uchar>(x, y)]++;

    // Passo 3: Normalizzazione dell'istogramma
    // Trasforma le frequenze in probabilità (somma = 1)
    double totalPixels = img.rows * img.cols;
    for (double &val : hist)
        val /= totalPixels;

    // Passo 4: Calcolo della media globale dell'immagine
    // μT = Σ (i * p(i)) per i = 0...255
    double globalMean = 0.0;
    for (int i = 0; i < 256; i++)
        globalMean += i * hist[i];

    // Passo 5: Ricerca esaustiva delle due soglie ottimali t1 e t2
    // Si esplorano tutte le combinazioni (t1, t2) tali che t1 < t2
    double maxVar = 0.0;
    int t1 = 0, t2 = 0;

    for (int i = 0; i < 254; i++) {  // t1
        double w0 = 0, m0 = 0;

        // Classe 1: livelli [0, t1]
        for (int k = 0; k <= i; k++) {
            w0 += hist[k];        // Peso (probabilità totale)
            m0 += k * hist[k];    // Somma pesata delle intensità
        }

        for (int j = i + 1; j < 255; j++) {  // t2
            double w1 = 0, m1 = 0;

            // Classe 2: livelli [t1+1, t2]
            for (int k = i + 1; k <= j; k++) {
                w1 += hist[k];
                m1 += k * hist[k];
            }

            // Classe 3: livelli [t2+1, 255]
            double w2 = 1.0 - w0 - w1;     // Peso della terza classe
            double m2 = globalMean - m0 - m1;  // Somma pesata residua

            // Calcolo della varianza inter-classe
            // σ_B² = Σ wi * (μi - μT)² per i = 1..3
            if (w0 > 0 && w1 > 0 && w2 > 0) {
                double var = w0 * pow(m0 / w0 - globalMean, 2) +
                             w1 * pow(m1 / w1 - globalMean, 2) +
                             w2 * pow(m2 / w2 - globalMean, 2);

                // Aggiornamento dei valori ottimali se la varianza è maggiore
                if (var > maxVar) {
                    maxVar = var;
                    t1 = i;
                    t2 = j;
                }
            }
        }
    }

    // Passo 6: Applicazione delle soglie all'immagine
    // Classifica i pixel in tre classi in base alle soglie trovate
    Mat out = Mat::zeros(img.size(), CV_8U);
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++) {
            uchar pixel = img.at<uchar>(x, y);
            if (pixel >= t2)
                out.at<uchar>(x, y) = 255;  // Classe 3: regione più chiara
            else if (pixel >= t1)
                out.at<uchar>(x, y) = 127;  // Classe 2: regione intermedia
            // Altrimenti rimane 0 (classe 1: regione scura)
        }

    // Restituisce l'immagine segmentata in tre livelli
    return out;
}
