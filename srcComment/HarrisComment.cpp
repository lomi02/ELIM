#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica l'algoritmo di rilevamento angoli di Harris a un'immagine in input.
 *
 * Questa funzione esegue i seguenti passaggi:
 * 1. Calcola le derivate spaziali orizzontali e verticali usando operatori di Sobel
 * 2. Calcola i prodotti delle derivate e li sfoca con un filtro gaussiano
 * 3. Calcola la matrice di autocorrelazione e la risposta R di Harris
 * 4. Normalizza e applica una soglia per ottenere gli angoli più significativi
 * 5. Disegna dei cerchi sui punti rilevati nell'immagine originale
 *
 * @param input      Immagine in input (scala di grigi)
 * @param k          Costante di Harris (solitamente tra 0.04 e 0.06)
 * @param threshTH   Soglia per selezionare i punti angolari
 *
 * @return Immagine con gli angoli rilevati disegnati (cerchi neri)
 */
Mat harris(Mat &input, float k, int threshTH) {
    Mat img = input.clone();

    // Passo 1: Calcolo delle derivate spaziali (gradiente)
    Mat Dx, Dy;
    Sobel(img, Dx, CV_32F, 1, 0); // Derivata in x
    Sobel(img, Dy, CV_32F, 0, 1); // Derivata in y

    // Passo 2: Calcolo dei prodotti delle derivate
    Mat Dx2, Dy2, DxDy;
    multiply(Dx, Dx, Dx2);   // Dx²
    multiply(Dy, Dy, Dy2);   // Dy²
    multiply(Dx, Dy, DxDy);  // Dx*Dy

    // Passo 2b: Applicazione di un filtro gaussiano per la media locale
    GaussianBlur(Dx2, Dx2, Size(3, 3), 0.5, 0.5);
    GaussianBlur(Dy2, Dy2, Size(3, 3), 0.5, 0.5);
    GaussianBlur(DxDy, DxDy, Size(3, 3), 0.5, 0.5);

    // Passo 3: Calcolo della risposta R di Harris
    Mat det = Dx2.mul(Dy2) - DxDy.mul(DxDy);  // Determinante della matrice di autocorrelazione
    Mat trace = Dx2 + Dy2;                    // Traccia della matrice di autocorrelazione
    Mat R = det - k * trace.mul(trace);       // Formula di Harris: R = det(M) - k * (trace(M))²

    // Passo 4: Normalizzazione e sogliatura
    normalize(R, R, 0, 255, NORM_MINMAX, CV_8U);   // Normalizza in intervallo [0,255]
    threshold(R, R, threshTH, 255, THRESH_BINARY); // Applica soglia per selezionare gli angoli

    // Passo 5: Disegno dei cerchi sui punti rilevati
    Mat out = input.clone();
    for (int x = 0; x < R.rows; x++)
        for (int y = 0; y < R.cols; y++)
            if (R.at<uchar>(x, y) > 0)
                circle(out, Point(y, x), 3, Scalar(0)); // Cerchio nero di raggio 3

    return out;
}
