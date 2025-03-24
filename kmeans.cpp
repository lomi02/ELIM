#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat kmeans(Mat &input, int k, int maxIterations, double convergenceThreshold) {
    Mat img = input.clone();

    // 1. Inizializzazione casuale dei centroidi
    srand(time(nullptr) + 1);
    vector<uchar> centersIntensity(k);
    for (int i = 0; i < k; i++) {
        int x = rand() % img.rows;
        int y = rand() % img.cols;
        centersIntensity.at(i) = img.at<uchar>(x, y);
    }

    // 2. Parametri
    int iterations = 0;
    int closestClusterIndex = 0;
    bool isCentreUpdated = true;
    vector<vector<Point> > clusters(k);
    Mat clusteredImg = img.clone();

    // 3. Inizio del ciclo iterativo K-means
    while (isCentreUpdated || iterations < maxIterations) {
        isCentreUpdated = false;
        for (int i = 0; i < k; i++)
            clusters.at(i).clear();

        // 4. Assegnazione di ogni pixel al cluster più vicino
        for (int x = 0; x < img.rows; x++)
            for (int y = 0; y < img.cols; y++) {
                uchar pixelIntensity = img.at<uchar>(x, y);
                int minDistance = INFINITY;

                for (int i = 0; i < k; i++) {
                    int currDistance = abs(centersIntensity.at(i) - pixelIntensity);
                    if (currDistance < minDistance) {
                        minDistance = currDistance;
                        closestClusterIndex = i;
                    }
                }

                // 5. Aggiunta del punto al cluster più vicino
                clusters.at(closestClusterIndex).emplace_back(y, x);
                clusteredImg.at<uchar>(x, y) = centersIntensity.at(closestClusterIndex);
            }

        // 6. Aggiornamento dei centroidi
        for (int i = 0; i < k; i++) {
            if (!clusters.at(i).empty()) {
                int intensitySum = 0;
                for (auto &point: clusters.at(i))
                    intensitySum += img.at<uchar>(point);

                int newCenter = intensitySum / clusters.at(i).size();
                int change = abs(newCenter - centersIntensity.at(i));

                // 8. Controllo della convergenza
                if (change > convergenceThreshold) {
                    centersIntensity.at(i) = newCenter;
                    isCentreUpdated = true;
                }
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

    Mat dst = kmeans(src, k, maxIterations, deltaTH);

    imshow("K-means", dst);
    waitKey(0);

    return 0;
}
