#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica l'algoritmo di rilevamento angoli di Harris a un'immagine in input.
 *
 * Questa funzione esegue i seguenti passaggi:
 * 1. Calcola le derivate orizzontali e verticali usando operatori di Sobel
 * 2. Calcola i prodotti delle derivate e i loro quadrati
 * 3. Applica uno smoothing gaussiano alle immagini delle derivate
 * 4. Calcola gli elementi del tensore strutturale
 * 5. Calcola la funzione di risposta di Harris R
 * 6. Applica una soglia per rilevare gli angoli
 * 7. Disegna cerchi sull'immagine nelle posizioni degli angoli rilevati
 *
 * @param input     Immagine in input (scala di grigi)
 * @param k         Parametro del rilevatore di Harris (tipicamente tra 0.04 e 0.06)
 * @param sobelSize Dimensione del kernel Sobel per il calcolo delle derivate
 * @param threshTH  Soglia per il rilevamento degli angoli (default 70)
 * @param blurSize  Dimensione del kernel gaussiano per lo smoothing (default 3)
 * @param blurSigma Deviazione standard per lo sfocamento gaussiano (default 0.5)
 *
 * @return Immagine con gli angoli rilevati marcati da cerchi
 */
Mat harris(Mat & input, float k, int sobelSize, int threshTH = 70, int blurSize = 3, float blurSigma = 0.5) {
    Mat img = input.clone();

    // Passo 1: Calcolo delle derivate orizzontali (Ix) e verticali (Iy)
    // Usiamo l'operatore di Sobel per stimare le derivate spaziali
    Mat Ix, Iy;
    Sobel(img, Ix, CV_32F, 1, 0, sobelSize);    // Derivata in x (orizzontale)
    Sobel(img, Iy, CV_32F, 0, 1, sobelSize);    // Derivata in y (verticale)

    // Passo 2: Calcolo dei prodotti delle derivate e dei loro quadrati
    // Questi termini sono necessari per costruire il tensore strutturale
    Mat IxIy;   // Prodotto Ix*Iy
    multiply(Ix, Iy, IxIy);

    pow(Ix, 2, Ix);     // Quadrato di Ix (Ix²)
    pow(Iy, 2, Iy);     // Quadrato di Iy (Iy²)

    // Passo 3: Smoothing gaussiano delle immagini delle derivate
    // Lo smoothing integra le informazioni su un intorno di ogni pixel
    GaussianBlur(Ix, Ix, Size(blurSize, blurSize), blurSigma, blurSigma);
    GaussianBlur(Iy, Iy, Size(blurSize, blurSize), blurSigma, blurSigma);
    GaussianBlur(IxIy, IxIy, Size(blurSize, blurSize), blurSigma, blurSigma);

    // Passo 4: Calcolo degli elementi del tensore strutturale M
    // M = [Ix²   IxIy]
    //     [IxIy   Iy²]
    Mat detM;   // Determinante di M (Ix²*Iy² - (IxIy)²)
    multiply(Ix, Iy, detM);
    Mat IxIy_squared;
    pow(IxIy, 2, IxIy_squared);
    detM -= IxIy_squared;

    Mat traceM;     // Traccia di M (Ix² + Iy²)
    pow(Ix + Iy, 2, traceM);

    // Passo 5: Calcolo della funzione di risposta di Harris R
    // R = det(M) - k*trace(M)²
    // Valori alti di R indicano la presenza di un angolo
    Mat harrisResponse = detM - k * traceM;

    // Normalizzazione della risposta nell'intervallo 0-255
    normalize(harrisResponse, harrisResponse, 0, 255, NORM_MINMAX, CV_8U);

    // Passo 6: Sogliatura per identificare gli angoli
    // I pixel con valore superiore alla soglia sono considerati angoli
    threshold(harrisResponse, harrisResponse, threshTH, 255, THRESH_BINARY);

    // Passo 7: Disegno dei cerchi nelle posizioni degli angoli rilevati
    Mat out = input.clone();
    for (int y = 0; y < harrisResponse.rows; ++y)
        for (int x = 0; x < harrisResponse.cols; ++x)
            if (harrisResponse.at<uchar>(Point(x, y)) > 0)

                // Disegna un cerchio di raggio 3 pixel in posizione (x, y)
                circle(out, Point(x, y), 3, Scalar(255), 1, 8, 0);

    return out;
}
