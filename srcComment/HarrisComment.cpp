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
Mat harris(Mat &input, float k, int sobelSize, int threshTH, int blurSize, float blurSigma) {
    Mat img = input.clone();

    // Passo 1: Calcolo delle derivate spaziali usando Sobel
    // Dx = derivata orizzontale (rispetto a x)
    // Dy = derivata verticale (rispetto a y)
    Mat Dx, Dy;
    Sobel(img, Dx, CV_32F, 1, 0, sobelSize);  // Derivata in x (CV_32F per precisione floating point)
    Sobel(img, Dy, CV_32F, 0, 1, sobelSize);  // Derivata in y

    // Passo 2: Calcolo dei prodotti delle derivate per il tensore strutturale
    // Dx² = quadrato della derivata in x
    // Dy² = quadrato della derivata in y
    // DxDy = prodotto delle derivate in x e y
    Mat Dx2, Dy2, DxDy;
    multiply(Dx, Dx, Dx2);  // Calcola Dx²
    multiply(Dy, Dy, Dy2);  // Calcola Dy²
    multiply(Dx, Dy, DxDy); // Calcola Dx*Dy

    // Passo 3: Smoothing gaussiano delle derivate
    // Applica un filtro gaussiano per integrare le informazioni su un intorno
    GaussianBlur(Dx2, Dx2, Size(blurSize, blurSize), blurSigma, blurSigma);     // Smoothing di Dx²
    GaussianBlur(Dy2, Dy2, Size(blurSize, blurSize), blurSigma, blurSigma);     // Smoothing di Dy²
    GaussianBlur(DxDy, DxDy, Size(blurSize, blurSize), blurSigma, blurSigma);   // Smoothing di DxDy

    // Passo 4: Calcolo del determinante e traccia del tensore strutturale
    // det(M) = Dx² * Dy² - (DxDy)²
    // trace(M) = Dx² + Dy²
    Mat det = Dx2.mul(Dy2) - DxDy.mul(DxDy);    // Calcola il determinante
    Mat trace = Dx2 + Dy2;                      // Calcola la traccia

    // Passo 5: Calcolo della risposta di Harris
    // R = det(M) - k * trace(M)²
    // Valori alti di R indicano la presenza di un angolo
    Mat R = det - k * trace.mul(trace);

    // Normalizzazione della risposta nell'intervallo 0-255
    // Converti in formato 8-bit per l'accesso ai pixel
    normalize(R, R, 0, 255, NORM_MINMAX, CV_8U);

    // Passo 6: Sogliatura per identificare gli angoli
    // I pixel con valore superiore alla soglia sono considerati angoli
    threshold(R, R, threshTH, 255, THRESH_BINARY);

    // Passo 7: Disegno dei cerchi nelle posizioni degli angoli rilevati
    Mat out = input.clone();
    for (int y = 0; y < R.rows; y++)
        for (int x = 0; x < R.cols; x++)
            if (R.at<uchar>(y, x) > 0)  // Se è un angolo

                // Disegna un cerchio nero di raggio 3 pixel
                circle(out, Point(x, y), 3, Scalar(0), 1, 8, 0);

    return out;
}
