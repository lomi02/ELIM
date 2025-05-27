#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica l'algoritmo di rilevamento bordi di Canny a un'immagine in input.
 *
 * Questa funzione esegue i seguenti passaggi:
 * 1. Applica uno smoothing gaussiano per ridurre il rumore
 * 2. Calcola le derivate orizzontali e verticali usando operatori di Sobel
 * 3. Calcola la magnitudo e la direzione del gradiente
 * 4. Applica la soppressione dei non-massimi per assottigliare i bordi
 * 5. Applica una sogliatura con isteresi usando due soglie
 *
 * @param input     Immagine in input (scala di grigi)
 * @param cannyLTH  Soglia inferiore per l'isteresi (valori tipici 5-30)
 * @param cannyHTH  Soglia superiore per l'isteresi (valori tipici 30-150)
 * @param blurSize  Dimensione del kernel gaussiano (default 3)
 * @param blurSigma Deviazione standard per lo sfocamento (default 0.5)
 *
 * @return Immagine binaria con i bordi rilevati (255=bordo, 0=sfondo)
 */
Mat canny(Mat &input, int cannyLTH, int cannyHTH, int blurSize, int blurSigma) {

    // Passo 1: Pre-elaborazione - Riduzione del rumore con filtro gaussiano
    Mat img = input.clone();
    GaussianBlur(img, img, Size(blurSize, blurSize), blurSigma, blurSigma);

    // Passo 2: Calcolo del gradiente
    Mat x_gradient, y_gradient;

    // Calcolo delle derivate nelle direzioni x e y usando l'operatore di Sobel
    // CV_32F indica che il risultato è in floating point per maggiore precisione
    Sobel(img, x_gradient, CV_32F, 1, 0);
    Sobel(img, y_gradient, CV_32F, 0, 1);

    // Calcolo della magnitudo
    Mat x_gradient2, y_gradient2, magnitude;
    pow(x_gradient, 2, x_gradient2);
    pow(y_gradient, 2, y_gradient2);
    sqrt(x_gradient2 + y_gradient2, magnitude);

    // Normalizzazione della magnitudo nell'intervallo 0-255
    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8U);

    // Calcolo della fase (direzione) del gradiente in gradi (0-360°)
    Mat phase;
    cv::phase(x_gradient, y_gradient, phase);

    // Passo 3: Soppressione dei non-massimi (NMS)
    Mat NMS = Mat::zeros(magnitude.size(), CV_8U);

    // Scansione dell'immagine (escludendo i bordi di 1 pixel)
    for (int x = 1; x < magnitude.rows - 1; x++)
        for (int y = 1; y < magnitude.cols - 1; y++) {

            // Normalizza l'angolo del gradiente in [0,180] con offset 22.5° per discretizzare
            // le 4 direzioni principali (0°, 45°, 90°, 135°) usate nella soppressione non-massima
            float angle = phase.at<float>(x, y);
            angle = fmod(angle + 22.5, 180);

            uchar curr = magnitude.at<uchar>(x, y);
            uchar pixel1, pixel2;

            // Determinazione della direzione del gradiente e selezione dei pixel adiacenti
            if (angle < 45) {

                // Direzione orizzontale
                pixel1 = magnitude.at<uchar>(x + 1, y);
                pixel2 = magnitude.at<uchar>(x - 1, y);

            } else if (angle < 90) {

                // Direzione diagonale (+45°)
                pixel1 = magnitude.at<uchar>(x + 1, y - 1);
                pixel2 = magnitude.at<uchar>(x - 1, y + 1);

            } else if (angle < 135) {

                // Direzione verticale
                pixel1 = magnitude.at<uchar>(x, y + 1);
                pixel2 = magnitude.at<uchar>(x, y - 1);

            } else {

                // Direzione diagonale (-45°)
                pixel1 = magnitude.at<uchar>(x + 1, y + 1);
                pixel2 = magnitude.at<uchar>(x - 1, y - 1);
            }

            // Mantiene solo i massimi locali nella direzione del gradiente
            if (curr >= pixel1 && curr >= pixel2)
                NMS.at<uchar>(x, y) = curr;
        }

    // Passo 4: Sogliatura con isteresi
    Mat edges = Mat::zeros(NMS.size(), CV_8U);

    // Scansione completa dell'immagine
    for (int x = 0; x < NMS.rows; x++)
        for (int y = 0; y < NMS.cols; y++) {
            uchar val = NMS.at<uchar>(x, y);

            // Se il pixel supera la soglia alta, è un bordo forte
            if (val >= cannyLTH && val <= cannyHTH)
                edges.at<uchar>(x, y) = 255;
        }

    return edges;
}