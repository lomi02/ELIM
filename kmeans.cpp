#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void kmeans(const Mat &src, Mat &dst, int k) {
    srand(time(nullptr));

    // 1. Inizializzazione casuale dei centroidi
    vector<uchar> centroids(k);
    for (int i = 0; i < k; ++i) {
        int randRow = rand() % src.rows;
        int randCol = rand() % src.cols;
        centroids[i] = src.at<uchar>(randRow, randCol);
    }

    // 2. Parametri di convergenza
    const double convergenceThreshold = 0.01;
    const int maxIterations = 50;
    vector<double> oldCentroids(k, 0.0);
    vector<double> newCentroids(k, 0.0);
    vector<vector<Point> > clusters(k);
    bool isCentroidsChanged = true;
    int iteration = 0;

    while (isCentroidsChanged && iteration++ < maxIterations) {
        isCentroidsChanged = false;

        // 3. Pulizia dei cluster
        for (int i = 0; i < k; ++i)
            clusters[i].clear();

        // 4. Salvataggio dello stato precedente dei centroidi
        for (int i = 0; i < k; ++i)
            oldCentroids[i] = newCentroids[i];

        // 5. Assegnazione di ciascun pixel al centroide più vicino
        for (int x = 0; x < src.rows; ++x)
            for (int y = 0; y < src.cols; ++y) {
                uchar pixelValue = src.at<uchar>(x, y);
                int bestCluster = 0;
                int minDistance = abs(pixelValue - centroids[0]);

                for (int i = 1; i < k; ++i) {
                    int distance = abs(pixelValue - centroids[i]);
                    if (distance < minDistance) {
                        minDistance = distance;
                        bestCluster = i;
                    }
                }
                clusters[bestCluster].push_back(Point(x, y));
            }

        // 6. Aggiornamento dei centroidi
        for (int i = 0; i < k; ++i) {
            int clusterSize = clusters[i].size();
            if (clusterSize > 0) {
                double sum = 0.0;
                for (int j = 0; j < clusterSize; ++j)
                    sum += src.at<uchar>(clusters[i][j].x, clusters[i][j].y);
                newCentroids[i] = sum / clusterSize;
            } else
                newCentroids[i] = centroids[i];
        }

        // 7. Controllo della convergenza
        for (int i = 0; i < k; ++i) {
            if (abs(newCentroids[i] - oldCentroids[i]) > convergenceThreshold)
                isCentroidsChanged = true;
            centroids[i] = static_cast<uchar>(newCentroids[i]);
        }

        cout << "Iteration: " << iteration << endl;
    }

    // 8. Creazione dell'immagine segmentata
    dst = src.clone();
    for (int i = 0; i < k; ++i) {
        int clusterSize = clusters[i].size();
        for (int j = 0; j < clusterSize; ++j)
            dst.at<uchar>(clusters[i][j].x, clusters[i][j].y) = centroids[i];
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/splash.png";
    Mat dst, src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    kmeans(src, dst, 3);

    imshow("K-means", dst);
    waitKey(0);

    return 0;
}
