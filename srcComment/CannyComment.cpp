#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica l'algoritmo di rilevamento bordi di Canny a un'immagine in input.
 *
 * Questa funzione esegue i seguenti passaggi:
 * 1. Sfocatura gaussiana per ridurre il rumore
 * 2. Calcolo del gradiente (magnitudo e fase)
 * 3. Soppressione dei non-massimi per mantenere solo i massimi locali del gradiente
 * 4. Sogliatura con isteresi per identificare i bordi usando soglie alta e bassa
 *
 * @param input     Immagine in input (scala di grigi)
 * @param cannyTHL  Soglia inferiore per la sogliatura con isteresi
 * @param cannyTHH  Soglia superiore per la sogliatura con isteresi
 * @param blurSize  Dimensione del kernel gaussiano (default 3)
 * @param blurSigma Deviazione standard per lo sfocamento gaussiano (default 0.5)
 *
 * @return Immagine binaria con i bordi rilevati (bordi=255, sfondo=0)
 */
Mat canny(Mat &input, int cannyTHL, int cannyTHH, int blurSize = 3, float blurSigma = 0.5) {
    Mat img = input.clone();

    // Passo 1: Applicazione dello sfocamento gaussiano per ridurre il rumore
    // Lo smoothing è fondamentale per eliminare rumore e falsi bordi
    GaussianBlur(img, img, Size(blurSize, blurSize), blurSigma, blurSigma);

    // Passo 2: Calcolo del gradiente, magnitudo e fase
    Mat x_gradient, y_gradient;

    // Calcola le derivate nelle direzioni x e y usando Sobel
    Sobel(img, x_gradient, CV_32F, 1, 0);
    Sobel(img, y_gradient, CV_32F, 0, 1);

    // Calcola la magnitudo approssimata come somma dei valori assoluti
    Mat magnitude = abs(x_gradient) + abs(y_gradient);

    // Normalizza la magnitudo nell'intervallo 0-255
    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8U);

    // Calcola la fase (direzione) del gradiente in radianti (0-2π)
    Mat phase;
    cv::phase(x_gradient, y_gradient, phase);

    // Passo 3: Soppressione dei non-massimi per assottigliare i bordi
    Mat NMS = magnitude.clone();
    uchar pixel1, pixel2;

    // Scansiona tutta l'immagine escludendo i bordi
    for (int y = 1; y < magnitude.rows - 1; ++y)
        for (int x = 1; x < magnitude.cols - 1; ++x) {
            float angle = phase.at<float>(Point(x, y));

            // Determina la direzione del gradiente e seleziona i pixel adiacenti per il confronto nella direzione perpendicolare al bordo
            if ((angle >= 360 - 22.5 && angle <= 22.5)
            || (angle >= 360 - 22.5 + 180 && angle <= 22.5 + 180)) {

                // Direzione orizzontale - confronta pixel a sinistra e destra
                pixel1 = magnitude.at<uchar>(Point(x + 1, y));
                pixel2 = magnitude.at<uchar>(Point(x - 1, y));

            } else if ((angle >= 22.5 && angle <= 22.5 + 45)
            || (angle >= 22.5 + 180 && angle <= 22.5 + 45 + 180)) {

                // Direzione diagonale 45° - confronta pixel nelle diagonali opposte
                pixel1 = magnitude.at<uchar>(Point(x - 1, y + 1));
                pixel2 = magnitude.at<uchar>(Point(x + 1, y - 1));

            } else if ((angle >= 22.5 + 45 && angle <= 22.5 + 90)
            || (angle >= 22.5 + 45 + 180 && angle <= 22.5 + 90 + 180)) {

                // Direzione verticale - confronta pixel sopra e sotto
                pixel1 = magnitude.at<uchar>(Point(x, y + 1));
                pixel2 = magnitude.at<uchar>(Point(x, y - 1));

            } else {

                // Direzione diagonale 135° - confronta pixel nelle diagonali opposte
                pixel1 = magnitude.at<uchar>(Point(x + 1, y - 1));
                pixel2 = magnitude.at<uchar>(Point(x - 1, y + 1));
            }

            // Sopprime il pixel se non è il massimo locale nella direzione del gradiente
            uchar currentMagnitude = magnitude.at<uchar>(Point(x, y));
            if (currentMagnitude < pixel1 || currentMagnitude < pixel2)
                NMS.at<uchar>(Point(x, y)) = 0;
        }

    // Passo 4: Sogliatura con isteresi per identificare i bordi forti e deboli
    Mat edgesImg = Mat::zeros(NMS.rows, NMS.cols, NMS.type());
    for (int y = 0; y < NMS.rows; ++y)
        for (int x = 0; x < NMS.cols; ++x)

            // Se il pixel supera la soglia alta, è un bordo forte
            if (NMS.at<uchar>(Point(x, y)) > cannyTHH) {
                edgesImg.at<uchar>(Point(x, y)) = 255;

                // Analisi dell'intorno 3x3 per trovare bordi deboli collegati
                Rect roi(x - 1, y - 1, 3, 3);
                for (int roi_y = roi.y; roi_y < roi.y + roi.height; ++roi_y)
                    for (int roi_x = roi.x; roi_x < roi.x + roi.width; ++roi_x)

                        // Se il pixel è tra le due soglie e collegato a un bordo forte, viene mantenuto
                        if (NMS.at<uchar>(Point(roi_x, roi_y)) > cannyTHL
                            && NMS.at<uchar>(Point(roi_x, roi_y)) < cannyTHH)
                            edgesImg.at<uchar>(Point(roi_x, roi_y)) = 255;
            }

    return edgesImg;
}
