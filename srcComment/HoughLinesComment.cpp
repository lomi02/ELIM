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
 * @param cannyTHL  Soglia inferiore per il rilevamento bordi di Canny
 * @param cannyTHH  Soglia superiore per il rilevamento bordi di Canny
 * @param blurSize  Dimensione del kernel gaussiano per lo smoothing
 * @param blurSigma Deviazione standard per lo sfocamento gaussiano
 *
 * @return Immagine con le linee rilevate disegnate
 */
Mat hough_lines(Mat &input, int houghTH, int cannyTHL, int cannyTHH, int blurSize, float blurSigma) {
    Mat img = input.clone();

    // Passo 1: Applicazione dello sfocamento gaussiano per ridurre il rumore
    // Lo smoothing è fondamentale per migliorare il rilevamento dei bordi successivo
    GaussianBlur(img, img, Size(blurSize, blurSize), blurSigma, blurSigma);

    // Passo 2: Rilevamento dei bordi con l'algoritmo di Canny
    // Canny trova i bordi significativi usando le due soglie (bassa e alta)
    Canny(img, img, cannyTHL, cannyTHH);

    // Passo 3: Calcolo della Trasformata di Hough per rilevare le linee
    // Calcoliamo la lunghezza diagonale dell'immagine per dimensionare lo spazio di Hough
    int diagonalLength = cvRound(hypot(img.rows, img.cols));

    // Matrice dei voti (accumulatore): righe=rho, colonne=theta
    Mat votes = Mat::zeros(diagonalLength * 2, 180, CV_8U);

    // Per ogni pixel dell'immagine...
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)

            // Se è un pixel di bordo (valore 255)...
                if (img.at<uchar>(x, y) == 255)

                // Per ogni possibile angolo theta...
                for (int theta = 0; theta < 180; theta++) {

                    // Calcola rho (distanza dall'origine) e aggiungi un voto
                    double thetaRad = theta * CV_PI / 180.0;
                    int rho = cvRound(x * sin(thetaRad) + y * cos(thetaRad));
                    int rhoIndex = rho + diagonalLength;    // Spostamento per evitare indici negativi
                    votes.at<uchar>(rhoIndex, theta)++;
                }

    // Passo 4: Disegna le linee rilevate sull'immagine originale
    Mat out = input.clone();
    int lineLength = max(img.rows, img.cols);   // Offset per estendere le linee

    // Itera su tutti i possibili valori di rho e theta
    for (int rhoIndex = 0; rhoIndex < votes.rows; rhoIndex++)
        for (int theta = 0; theta < votes.cols; theta++)

            // Se i voti superano la soglia, abbiamo trovato una linea
            if (votes.at<uchar>(rhoIndex, theta) > houghTH) {
                int rho = rhoIndex - diagonalLength;    // Ripristina il valore originale di rho
                double thetaRad = theta * CV_PI / 180.0;

                // Calcola le coordinate di due punti per disegnare la linea
                double a = cos(thetaRad), b = sin(thetaRad);
                double x0 = a * rho, y0 = b * rho;

                // Primo punto della linea
                Point point1;
                point1.x = cvRound(x0 + lineLength * -b);
                point1.y = cvRound(y0 + lineLength * a);

                // Secondo punto della linea
                Point point2;
                point2.x = cvRound(x0 - lineLength * -b);
                point2.y = cvRound(y0 - lineLength * a);

                // Disegna la linea sull'immagine
                line(out, point1, point2, Scalar(0), 2, 0);
            }

    return out;
}
