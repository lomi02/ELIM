#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

vector<double> normalizedHistogram(Mat &src) {

    // Inizializzo l'istogramma a 256 slot (Valori grayscale 0 - 255)
    vector<double> hist(256);

    // Popolo l'istogramma iterando ogni pixel dell'immagine
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)

            // Incremento il conteggio dell'intensità del pixel corrente
            hist[src.at<uchar>(i, j)]++;

    double totalPixels = src.rows * src.cols;

    // Normalizzo l'istogramma dividendo ciascun counter d'intensità di pixel con il numero totale di pixel
    for (int i = 0; i < 256; i++)
        hist[i] /= totalPixels;

    return hist;
}

vector<int> otsu2k(Mat &src) {

    // Ottengo l'istogramma normalizzato dell'immagine
    vector<double> hist = normalizedHistogram(src);

    // Calcolo la media globale dell'intensità dell'immagine
    double globalMean = 0;
    for (int i = 0; i < 256; i++)
        globalMean += i * hist[i];

    double maxVar = 0; 		// Variabile per immagazzinare la varianza interclasse massima
    vector<int> kstar(2); 	// Vettore per contenere le due soglie ottimali

    // Distribuzione della probabilità cumulativa
    vector<double> cumProb(256);
    cumProb[0] = hist[0];

    // Distribuzione della media cumulativa
    vector<double> cumMean(256);
    cumMean[0] = 0;

    // Calcolo probabilità e media cumulativa per tutti i livelli di intensità
    for (int i = 1; i < 256; i++) {
        cumProb[i] = cumProb[i - 1] + hist[i];
        cumMean[i] = cumMean[i - 1] + i * hist[i];
    }

    // Itero su tutte le possibili coppie di soglie (i, j)
    for (int i = 0; i < 256 - 2; i++)
        for (int j = i + 1; j < 256 - 1; j++) {

            // Calcolo della classe 1: Intensità < i
            double prob1 = cumProb[i];
            double mean1 = cumMean[i] / prob1;

            // Calcolo della classe 2: i < Intensità < j
            double prob2 = cumProb[j] - cumProb[i];
            double mean2 = (cumMean[j] - cumMean[i]) / prob2;

            // Calcolo della classe 3: Intensità > j
            double prob3 = cumProb[255] - cumProb[j];
            double mean3 = (cumMean[255] - cumMean[j]) / prob3;

            // Compila la varianza interclasse per tutte le soglie correnti
            double interClassVar = prob1 * (mean1 - globalMean) * (mean1 - globalMean) +
                                   prob2 * (mean2 - globalMean) * (mean2 - globalMean) +
                                   prob3 * (mean3 - globalMean) * (mean3 - globalMean);

            // Aggiorna le soglie ottimali se la varianza corrente è più grande di quella massima
            if (interClassVar > maxVar) {
                maxVar = interClassVar;
                kstar[0] = i;
                kstar[1] = j;
            }
        }

    // Restituisci le soglie ottimali
    return kstar;
}

void multipleThresholds(Mat &src, Mat &dst, int th1, int th2) {
    dst = Mat::zeros(src.rows, src.cols, CV_8U);

    // Itera su ogni pixel dell'immagine
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++) {

            // Ottieni l'intensità del pixel
            uchar pixel = src.at<uchar>(i, j);

            // Se il pixel è maggiore della soglia alta, questo diventa bianco (255)
            if (pixel >= th2)
                dst.at<uchar>(i, j) = 255;

            // Se il pixel è maggiore della soglia bassa, questo diventa grigio (127)
            if (pixel >= th1 && pixel < th2)
                dst.at<uchar>(i, j) = 127;

            // Se il pixel è minore della soglia bassa, rimane nero (0)
        }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/splash.png";
    Mat dst, src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    Mat gauss;
    GaussianBlur(src, gauss, Size(3, 3), 0);

    auto thresholds = otsu2k(gauss);
    multipleThresholds(gauss, dst, thresholds[0], thresholds[1]);

    imshow("Otsu2k", dst);
    waitKey(0);

    return 0;
}