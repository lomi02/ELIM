#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica l'algoritmo di clustering k-means a un'immagine in scala di grigi.
 *
 * @param input    Immagine in scala di grigi in input da clusterizzare.
 * @param k        Numero di cluster da creare.
 * @return         Immagine clusterizzata in scala di grigi.
 */
Mat kmeans(Mat &input, int k) {
    Mat img = input.clone();
    srand(time(nullptr));

    // Inizializza i centroidi in modo casuale selezionando pixel a caso
    vector<uchar> centroids(k);
    for (int i = 0; i < k; i++) {
        int x = rand() % img.rows;
        int y = rand() % img.cols;
        centroids[i] = img.at<uchar>(x, y);
    }

    // Inizializza i cluster per contenere i punti assegnati a ciascun centroide
    vector<vector<Point> > clusters(k);

    // Ciclo principale: assegna i punti e aggiorna i centroidi
    for (int iter = 0; iter < 50; iter++) {

        // Pulisce tutti i cluster per la nuova iterazione
        for (auto &cluster: clusters)
            cluster.clear();

        // Assegna ogni pixel al centroide più vicino
        for (int x = 0; x < img.rows; x++)
            for (int y = 0; y < img.cols; y++) {
                uchar pixel = img.at<uchar>(x, y);

                // Trova il centroide più vicino al pixel corrente
                int best = 0;
                for (int i = 1; i < k; i++)
                    if (abs(centroids[i] - pixel) < abs(centroids[best] - pixel))
                        best = i;

                clusters[best].push_back(Point(x, y));
            }

        // Aggiorna i centroidi calcolando la media dei pixel nei cluster
        bool changed = false;
        for (int i = 0; i < k; i++) {
            if (clusters[i].empty())
                continue;

            // Calcola la somma dei valori dei pixel nel cluster corrente
            int sum = 0;
            for (Point &p: clusters[i])
                sum += img.at<uchar>(p.x, p.y);

            // Calcola il nuovo centroide come media e verifica se è cambiato
            uchar newCentroid = sum / clusters[i].size();
            if (abs(newCentroid - centroids[i]) > 0.01)
                changed = true;

            centroids[i] = newCentroid;
        }

        // Se nessun centroide è cambiato, l'algoritmo è converso
        if (!changed)
            break;
    }

    // Crea l'immagine di output con i pixel sostituiti dai valori dei centroidi
    Mat out = img.clone();
    for (int i = 0; i < k; i++)
        for (Point &p: clusters[i])
            out.at<uchar>(p.x, p.y) = centroids[i];

    return out;
}
