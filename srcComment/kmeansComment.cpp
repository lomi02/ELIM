#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica l'algoritmo di clustering k-means a un'immagine in scala di grigi.
 *
 * @param input             Immagine in scala di grigi in input da clusterizzare.
 * @param numberOfClusters  Numero di cluster da creare.
 * @param maxIterations     Numero massimo di iterazioni per l'algoritmo.
 * @param deltaTH          Soglia per l'aggiornamento dei centri per fermare le iterazioni.
 * @return                  Immagine clusterizzata in scala di grigi.
 */
Mat kmeans_gray(Mat &input, int numberOfClusters, int maxIterations, double deltaTH = 1.0) {
    Mat img = input.clone();

    // Passo 1: Inizializza i centri dei cluster in modo casuale.
    srand(time(nullptr) + 1);
    vector<uchar> centres(numberOfClusters);
    for (int i = 0; i < numberOfClusters; ++i) {

        // Seleziona una posizione casuale nell'immagine
        int x = rand() % img.cols;
        int y = rand() % img.rows;

        // Assegna il valore di intensità del pixel casuale come centro del cluster
        centres.at(i) = img.at<uchar>(Point(x, y));
    }

    // Variabili per il controllo delle iterazioni
    int iterations = 0;
    int closestIndex = 0;
    bool isCentreUpdated = true;

    // Inizializza i contenitori per i cluster e l'immagine clusterizzata
    vector<vector<Point> > clusters(numberOfClusters);
    Mat clusteredImg = img.clone();

    // Continua le iterazioni finché i centri si aggiornano o finché non si raggiunge maxIterations
    while (isCentreUpdated or iterations < maxIterations) {

        // Resetta il flag di aggiornamento e i contenitori dei cluster
        isCentreUpdated = false;
        for (int i = 0; i < numberOfClusters; ++i)
            clusters.at(i).clear();

        // Passo 2: Assegna ogni pixel al cluster con il centro più vicino
        for (int y = 0; y < img.rows; ++y)
            for (int x = 0; x < img.cols; ++x) {
                int minDistance = INFINITY;

                // Trova il cluster con il centro più vicino al pixel corrente
                for (int i = 0; i < numberOfClusters; ++i) {
                    int currentDistance = abs(centres.at(i) - img.at<uchar>(Point(x, y)));
                    if (currentDistance < minDistance) {
                        minDistance = currentDistance;
                        closestIndex = i;
                    }
                }
                // Aggiungi il pixel al cluster selezionato e aggiorna l'immagine clusterizzata
                clusters.at(closestIndex).emplace_back(x, y);
                clusteredImg.at<uchar>(Point(x, y)) = centres.at(closestIndex);
            }

        // Passo 3: Calcola i nuovi centri come media dei pixel nei cluster
        for (int i = 0; i < numberOfClusters; ++i)
            if (not clusters.at(i).empty()) {
                int intensitySum = 0;

                // Calcola la somma delle intensità dei pixel nel cluster
                for (auto &point: clusters.at(i))
                    intensitySum += img.at<uchar>(point);

                // Calcola la nuova media e la differenza con il centro precedente
                double currentMean = intensitySum / clusters.at(i).size();
                int delta = cvRound(std::abs(currentMean - centres.at(i)));

                // Aggiorna il centro se la differenza supera la soglia
                if (delta > deltaTH) {
                    centres.at(i) = cvRound(currentMean);
                    isCentreUpdated = true;
                }
            }
        iterations++;
    }
    return clusteredImg;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/splash.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    int k = 3;
    int maxIterations = 30;
    double deltaTH = 1.0;

    Mat dst = kmeans_gray(src, k, maxIterations, deltaTH);

    imshow("K-means", dst);
    waitKey(0);

    return 0;
}
