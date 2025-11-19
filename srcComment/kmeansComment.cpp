#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica l'algoritmo K-means per segmentare un'immagine in scala di grigi in k cluster.
 *
 * Questa funzione esegue i seguenti passaggi:
 * 1. Inizializza casualmente k centroidi scegliendo pixel dell'immagine
 * 2. Assegna ogni pixel al cluster il cui centroide è più vicino (minima distanza in intensità)
 * 3. Aggiorna i centroidi calcolando la media dei pixel assegnati a ciascun cluster
 * 4. Ripete i passi 2-3 fino a convergenza o massimo numero di iterazioni
 * 5. Assegna a ciascun pixel il valore del centroide del cluster di appartenenza
 *
 * @param input  Immagine in input (scala di grigi)
 * @param k      Numero di cluster
 *
 * @return Immagine segmentata secondo i k cluster
 */
Mat kmeans(Mat &input, int k) {
    Mat img = input.clone();

    // Inizializzazione casuale
    srand(time(nullptr));

    // Passo 1: Inizializzazione dei centroidi scegliendo pixel casuali
    vector<uchar> centroids(k);
    for (int i = 0; i < k; i++) {
        int x = rand() % img.rows;
        int y = rand() % img.cols;
        centroids[i] = img.at<uchar>(x, y); // Valore del pixel come centroide iniziale
    }

    // Cluster: vettore di punti assegnati a ciascun centroide
    vector<vector<Point> > clusters(k);

    // Passo 2-4: Iterazioni principali dell'algoritmo K-means
    for (int iter = 0; iter < 50; iter++) { // Massimo 50 iterazioni
        // Svuota i cluster dall'iterazione precedente
        for (size_t i = 0; i < clusters.size(); i++)
            clusters[i].clear();

        // Assegnazione dei pixel al cluster più vicino
        for (int x = 0; x < img.rows; x++)
            for (int y = 0; y < img.cols; y++) {
                uchar pixel = img.at<uchar>(x, y);

                int best = 0;
                for (int i = 1; i < k; i++)
                    if (abs(centroids[i] - pixel) < abs(centroids[best] - pixel))
                        best = i; // Trova il centroide più vicino
                clusters[best].push_back(Point(x, y));
            }

        // Aggiornamento dei centroidi
        bool changed = false; // Flag per controllare convergenza
        for (int i = 0; i < k; i++) {
            if (clusters[i].empty())
                continue;

            int sum = 0;
            for (size_t j = 0; j < clusters[i].size(); j++)
                sum += img.at<uchar>(clusters[i][j].x, clusters[i][j].y);

            uchar newCentroid = sum / clusters[i].size(); // Media dei pixel
            if (newCentroid != centroids[i])
                changed = true; // Se cambia almeno un centroide, continua iterazioni
            centroids[i] = newCentroid;
        }

        if (!changed) // Convergenza raggiunta
            break;
    }

    // Passo 5: Assegna a ciascun pixel il valore del centroide del cluster
    Mat out = img.clone();
    for (int i = 0; i < k; i++)
        for (size_t j = 0; j < clusters[i].size(); j++)
            out.at<uchar>(clusters[i][j].x, clusters[i][j].y) = centroids[i];

    return out;
}
