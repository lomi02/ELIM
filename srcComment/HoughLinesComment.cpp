#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica l'algoritmo di rilevamento linee di Hough a un'immagine in input.
 *
 * Questa funzione esegue i seguenti passaggi:
 * 1. Applica uno sfocamento gaussiano per ridurre il rumore
 * 2. Esegue il rilevamento dei bordi con Canny
 * 3. Calcola la Trasformata di Hough per individuare le linee
 * 4. Disegna le linee rilevate sull'immagine originale
 *
 * @param input     Immagine in input (scala di grigi)
 * @param houghTH   Soglia per il rilevamento delle linee nello spazio di Hough
 *
 * @return Immagine con le linee rilevate disegnate
 */
Mat hough_lines(Mat &input, int houghTH) {
    Mat img = input.clone();

    // Passo 1: Applicazione dello sfocamento gaussiano per ridurre il rumore
    // Lo smoothing è fondamentale per migliorare il rilevamento dei bordi successivo
    // Usa un kernel fisso 5x5 con deviazione standard 0.5
    GaussianBlur(input, img, Size(5, 5), 0.5, 0.5);

    // Passo 2: Rilevamento dei bordi con l'algoritmo di Canny
    // Canny trova i bordi significativi usando soglie fisse (50 e 150)
    Canny(img, img, 50, 150);

    // Passo 3: Calcolo della Trasformata di Hough per rilevare le linee
    // Calcoliamo la lunghezza diagonale dell'immagine per dimensionare lo spazio di Hough
    int diag = cvRound(hypot(img.rows, img.cols));

    // Matrice dei voti (accumulatore): righe=rho (2*diag per evitare indici negativi), colonne=theta
    Mat votes = Mat::zeros(diag * 2, 180, CV_8U);

    // Per ogni pixel dell'immagine...
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)

            // Se è un pixel di bordo (valore 255)...
            if (img.at<uchar>(x, y) == 255)

                // Per ogni possibile angolo theta (in gradi)...
                for (int thetaDeg = 0; thetaDeg < 180; thetaDeg++) {

                    // Converti l'angolo in radianti
                    double thetaRad = thetaDeg * CV_PI / 180.0;

                    // Calcola rho (distanza dall'origine) e aggiungi un voto
                    int rho = cvRound(x * sin(thetaRad) + y * cos(thetaRad)) + diag;

                    // Incrementa l'accumulatore per questa coppia (rho, theta)
                    votes.at<uchar>(rho, thetaDeg)++;
                }

    // Passo 4: Disegna le linee rilevate sull'immagine originale
    Mat out = input.clone();
    int lineLength = max(img.rows, img.cols); // Lunghezza sufficiente per linee che attraversano l'intera immagine

    // Itera su tutti i possibili valori di rho e theta
    for (int rho = 0; rho < votes.rows; rho++)
        for (int thetaDeg = 0; thetaDeg < votes.cols; thetaDeg++)

            // Se i voti superano la soglia, abbiamo trovato una linea
            if (votes.at<uchar>(rho, thetaDeg) > houghTH) {
                // Converti l'angolo in radianti
                double thetaRad = thetaDeg * CV_PI / 180.0;

                // Calcola i coefficienti della retta
                double a = cos(thetaRad), b = sin(thetaRad);

                // Calcola un punto sulla retta (x0, y0)
                double x0 = a * (rho - diag);
                double y0 = b * (rho - diag);

                // Calcola i punti estremi della linea da disegnare
                Point p1(cvRound(x0 - lineLength * b), cvRound(y0 + lineLength * a));
                Point p2(cvRound(x0 + lineLength * b), cvRound(y0 - lineLength * a));

                // Disegna la linea sull'immagine
                line(out, p1, p2, Scalar(0), 2);
            }

    return out;
}
