#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica l'algoritmo di trasformata di Hough per rilevare linee in un'immagine in input.
 *
 * Questa funzione esegue i seguenti passaggi:
 * 1. Applica un filtro gaussiano per ridurre il rumore
 * 2. Esegue il rilevamento dei bordi tramite l'algoritmo di Canny
 * 3. Popola un accumulatore 2D (rho, theta) votando per le possibili linee
 * 4. Seleziona i valori con voti sufficienti e disegna le linee corrispondenti sull'immagine originale
 *
 * @param input     Immagine in input (scala di grigi)
 * @param houghTH   Soglia minima di voti nell'accumulatore per considerare una linea valida
 *
 * @return Immagine con le linee rilevate disegnate (linee nere)
 */
Mat hough_lines(Mat &input, int houghTH) {
    Mat img = input.clone();

    // Passo 1: Riduzione del rumore tramite filtro gaussiano 5x5
    GaussianBlur(img, img, Size(5, 5), 0.5, 0.5);

    // Passo 2: Rilevamento dei bordi con Canny
    Canny(img, img, 50, 150);

    // Passo 3: Preparazione dell'accumulatore 2D (rho x theta)
    int diag = cvRound(hypot(img.rows, img.cols));  // Lunghezza massima della diagonale
    Mat votes = Mat::zeros(diag * 2, 180, CV_8U);   // Rho in [-diag, diag], theta in gradi [0,179]

    // Ciclo su tutti i pixel dell'immagine dei bordi
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            if (img.at<uchar>(x, y) == 255) // Solo pixel di bordo
                for (int thetaDeg = 0; thetaDeg < 180; thetaDeg++) { // Ciclo su tutte le possibili direzioni
                    double thetaRad = thetaDeg * CV_PI / 180.0;
                    int rho = cvRound(x * sin(thetaRad) + y * cos(thetaRad)) + diag; // Traslazione per indice positivo

                    votes.at<uchar>(rho, thetaDeg)++; // Incrementa il voto nell'accumulatore
                }

    // Passo 4: Disegno delle linee con voti sufficienti sull'immagine originale
    Mat out = input.clone();
    int lineLength = max(img.rows, img.cols); // Lunghezza della linea da disegnare
    for (int rho = 0; rho < votes.rows; rho++)
        for (int thetaDeg = 0; thetaDeg < votes.cols; thetaDeg++)
            if (votes.at<uchar>(rho, thetaDeg) > houghTH) { // Se il voto supera la soglia
                double thetaRad = thetaDeg * CV_PI / 180.0;
                double a = cos(thetaRad), b = sin(thetaRad);
                double x0 = a * (rho - diag);
                double y0 = b * (rho - diag);

                // Calcolo dei due punti estremi della linea per disegno
                Point p1(cvRound(x0 - lineLength * b), cvRound(y0 + lineLength * a));
                Point p2(cvRound(x0 + lineLength * b), cvRound(y0 - lineLength * a));

                line(out, p1, p2, Scalar(0), 2); // Disegna la linea nera di spessore 2
            }

    return out;
}
