#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica l'algoritmo di rilevamento bordi di Canny a un'immagine in input.
 *
 * Questa funzione esegue i seguenti passaggi:
 * 1. Applica un filtro gaussiano per ridurre il rumore
 * 2. Calcola le derivate orizzontali e verticali usando operatori di Sobel
 * 3. Calcola la magnitudine e la direzione del gradiente
 * 4. Applica il Non-Maximum Suppression (NMS) per affinare i bordi
 * 5. Applica la soglia doppia per determinare i bordi finali
 *
 * @param input     Immagine in input (scala di grigi)
 * @param cannyLTH  Soglia bassa per la rilevazione dei bordi
 * @param cannyHTH  Soglia alta per la rilevazione dei bordi
 *
 * @return Immagine binaria con i bordi rilevati (255 = bordo, 0 = sfondo)
 */
Mat canny(Mat &input, int cannyLTH, int cannyHTH) {
    Mat img = input.clone();

    // Passo 1: Riduzione del rumore tramite filtro gaussiano 3x3
    GaussianBlur(img, img, Size(3, 3), 1, 1);

    // Passo 2: Calcolo delle derivate spaziali (gradiente)
    Mat Dx, Dy;
    Sobel(img, Dx, CV_32F, 1, 0);  // Derivata in x
    Sobel(img, Dy, CV_32F, 0, 1);  // Derivata in y

    // Passo 3: Calcolo della magnitudine del gradiente
    Mat Dx2, Dy2, magnitude;
    multiply(Dx, Dx, Dx2);          // Dx²
    multiply(Dy, Dy, Dy2);          // Dy²
    sqrt(Dx2 + Dy2, magnitude);     // sqrt(Dx² + Dy²)

    // Normalizzazione in intervallo 0-255 (8-bit)
    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8U);

    // Calcolo della direzione del gradiente (fase)
    Mat phase;
    cv::phase(Dx, Dy, phase);

    // Passo 4: Non-Maximum Suppression (NMS)
    // Mantiene solo i massimi locali lungo la direzione del gradiente
    Mat NMS = Mat::zeros(magnitude.size(), CV_8U);
    for (int x = 1; x < magnitude.rows; x++)
        for (int y = 1; y < magnitude.cols; y++) {
            float angle = phase.at<float>(x, y);
            angle = fmod(angle + 22.5, 180);  // Normalizza in intervallo [0,180)

            uchar curr = magnitude.at<uchar>(x, y);
            uchar pixel1, pixel2;

            // Determina i due pixel lungo la direzione del gradiente
            if (angle < 45) {
                pixel1 = magnitude.at<uchar>(x + 1, y);
                pixel2 = magnitude.at<uchar>(x - 1, y);
            } else if (angle < 90) {
                pixel1 = magnitude.at<uchar>(x + 1, y - 1);
                pixel2 = magnitude.at<uchar>(x - 1, y + 1);
            } else if (angle < 135) {
                pixel1 = magnitude.at<uchar>(x, y + 1);
                pixel2 = magnitude.at<uchar>(x, y - 1);
            } else {
                pixel1 = magnitude.at<uchar>(x + 1, y + 1);
                pixel2 = magnitude.at<uchar>(x - 1, y - 1);
            }

            // Mantiene il pixel solo se è massimo locale
            if (curr >= pixel1 && curr >= pixel2)
                NMS.at<uchar>(x, y) = curr;
        }

    // Passo 5: Soglia doppia (hysteresis)
    // Se la magnitudine è tra soglia bassa e alta, viene considerato bordo
    Mat out = Mat::zeros(NMS.size(), CV_8U);
    for (int x = 1; x < NMS.rows; x++)
        for (int y = 1; y < NMS.cols; y++)
            if (NMS.at<uchar>(x, y) > cannyLTH && NMS.at<uchar>(x, y) < cannyHTH) {
                out.at<uchar>(x, y) = 255;

                // Propagazione ai pixel vicini per connettere i bordi
                for (int nx = -1; nx <= 1; nx++)
                    for (int ny = -1; ny <= 1; ny++)
                        if (NMS.at<uchar>(x + nx, y + ny) > cannyLTH && NMS.at<uchar>(x + nx, y + ny) < cannyHTH)
                            out.at<uchar>(x + nx, y + ny) = 255;
            }

    return out;
}
