#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica l'algoritmo di clustering k-means a un'immagine in scala di grigi.
 *
 * @param input             Immagine in scala di grigi in input da clusterizzare.
 * @param numberOfCluster   Numero di cluster da creare.
 * @param maxIterations     Numero massimo di iterazioni per l'algoritmo.
 * @param deltaTH           Soglia per l'aggiornamento dei centri per fermare le iterazioni.
 * @return                  Immagine clusterizzata in scala di grigi.
 */
Mat kmeans_gray(Mat &input, int numberOfCluster, int maxIterations, double deltaTH) {
    Mat img = input.clone();

    // Passo 1: Inizializza i centri dei cluster in modo casuale.
    srand(time(nullptr) + 1);
    vector<uchar> centres(numberOfCluster);
    for (int i = 0; i < numberOfCluster; i++) {
        // Seleziona una posizione casuale nell'immagine
        int x = rand() % img.rows;
        int y = rand() % img.cols;

        // Assegna il valore di intensità del pixel casuale come centro del cluster
        centres.at(i) = img.at<uchar>(x, y);
    }

    // Variabili per il controllo delle iterazioni
    int iterations = 0;
    int closestIndex = 0;
    bool isCentreUpdated = true;

    // Inizializza i contenitori per i cluster e l'immagine clusterizzata
    vector<vector<Point> > clusters(numberOfCluster);
    Mat out = input.clone();

    // Continua le iterazioni finché i centri si aggiornano o finché non si raggiunge maxIterations
    while (isCentreUpdated && iterations < maxIterations) {
        // Resetta il flag di aggiornamento e i contenitori dei cluster
        isCentreUpdated = false;
        for (int i = 0; i < numberOfCluster; i++)
            clusters.at(i).clear();

        // Passo 2: Assegna ogni pixel al cluster con il centro più vicino
        for (int x = 0; x < img.rows; x++)
            for (int y = 0; y < img.cols; y++) {
                int minDistance = INT_MAX;

                // Trova il cluster con il centro più vicino al pixel corrente
                for (int i = 0; i < numberOfCluster; i++) {
                    int currentDistance = abs(centres.at(i) - img.at<uchar>(x, y));
                    if (currentDistance < minDistance) {
                        minDistance = currentDistance;
                        closestIndex = i;
                    }
                }
                // Aggiungi il pixel al cluster selezionato e aggiorna l'immagine clusterizzata
                clusters.at(closestIndex).emplace_back(x, y);
                out.at<uchar>(x, y) = centres.at(closestIndex);
            }

        // Passo 3: Calcola i nuovi centri come media dei pixel nei cluster
        for (int i = 0; i < numberOfCluster; i++)
            if (!clusters.at(i).empty()) {
                int intensitySum = 0;

                // Calcola la somma delle intensità dei pixel nel cluster
                for (auto &point: clusters.at(i))
                    intensitySum += img.at<uchar>(point);

                // Calcola la nuova media e la differenza con il centro precedente
                double currentMean = intensitySum / clusters.at(i).size();
                int delta = cvRound(abs(currentMean - centres.at(i)));

                // Aggiorna il centro se la differenza supera la soglia
                if (delta > deltaTH) {
                    centres.at(i) = cvRound(currentMean);
                    isCentreUpdated = true;
                }
            }
        iterations++;
    }

    return out;
}