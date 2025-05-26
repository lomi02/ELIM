#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica l'algoritmo di clustering k-means a un'immagine in scala di grigi.
 *
 * @param input           Immagine in scala di grigi in input da clusterizzare.
 * @param k               Numero di cluster da creare.
 * @param maxIterations   Numero massimo di iterazioni per l'algoritmo.
 * @param deltaTH         Soglia per l'aggiornamento dei centri per fermare le iterazioni.
 * @return                Immagine clusterizzata in scala di grigi.
 */
Mat kmeans(Mat &input, int k, int maxIterations, double deltaTH) {
    Mat img = input.clone();

    // Estrai tutti i pixel dell'immagine in un vettore unidimensionale
    vector<uchar> pixels;
    for (int i = 0; i < input.rows; i++)
        for (int j = 0; j < input.cols; j++)
            pixels.push_back(img.at<uchar>(i, j));

    // Inizializza i centri dei cluster in modo casuale selezionando pixel a caso
    vector<uchar> centroids(k);
    for (int i = 0; i < k; i++)
        centroids[i] = pixels[rand() % pixels.size()];

    // Inizializza le etichette dei cluster e i flag per il controllo delle iterazioni
    vector<int> labels(pixels.size());
    bool centroidsChanged = true;
    int iterations = 0;

    // Continua le iterazioni finché i centri si aggiornano o finché non si raggiunge maxIterations
    while (centroidsChanged && iterations < maxIterations) {
        centroidsChanged = false;

        // Assegna ogni pixel al cluster con il centro più vicino
        for (int i = 0; i < pixels.size(); i++) {
            int minDist = INT_MAX, nearestCluster = 0;

            // Trova il cluster con il centro più vicino al pixel corrente
            for (int cluster = 0; cluster < k; cluster++) {
                int dist = abs(pixels[i] - centroids[cluster]);

                if (dist < minDist) {
                    minDist = dist;
                    nearestCluster = cluster;
                }
            }
            labels[i] = nearestCluster;
        }

        // Calcola i nuovi centri come media dei pixel nei cluster
        for (int cluster = 0; cluster < k; cluster++) {
            double sum = 0, count = 0;

            // Calcola la somma e il conteggio dei pixel nel cluster corrente
            for (int i = 0; i < pixels.size(); i++) {
                if (labels[i] == cluster) {
                    sum += pixels[i];
                    count++;
                }
            }

            // Se il cluster non è vuoto, calcola il nuovo centro
            if (count > 0) {
                double newCenter = sum / count;

                // Aggiorna il centro se la differenza supera la soglia
                if (abs(newCenter - centroids[cluster]) > deltaTH) {
                    centroids[cluster] = newCenter;
                    centroidsChanged = true;
                }
            }
        }
        iterations++;
    }

    // Crea l'immagine di output con i pixel sostituiti dai valori dei centri
    Mat out(img.size(), CV_8U);
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            out.at<uchar>(i, j) = centroids[labels[i * img.cols + j]];

    return out;
}
