#include <opencv2/opencv.hpp>
#include <stack>
using namespace std;
using namespace cv;

// 1. Funzione per visualizzare le regioni segmentate
void printMasks(const Mat &src, uchar labelCount) {
    Mat temp_img = Mat::zeros(src.rows, src.cols, CV_8U);

    // 1.1. Mostra ogni regione segmentata separatamente
    for (uchar label = 1; label < labelCount; label++) {
        for (int i = 0; i < src.rows; i++)
            for (int j = 0; j < src.cols; j++)
                if (src.at<uchar>(i, j) == label)
                    temp_img.at<uchar>(i, j) = 255;

        imshow("Region " + to_string(label), temp_img);
        waitKey(0);
        temp_img.setTo(0);
    }

    // 1.2. Mostra i pixel "don't care"
    temp_img.setTo(0);
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
            if (src.at<uchar>(i, j) == 255)
                temp_img.at<uchar>(i, j) = 255;

    imshow("Don't Care", temp_img);
    waitKey(0);
}

// 2. Funzione del region growing
void grow(const Mat &src, Mat &dst, Mat &mask, Point seed, int th, int &area) {
    const Point shift8D[8] = {
        Point(-1, -1), Point(-1, 0), Point(-1, 1),
        Point(0, -1), Point(0, 1),
        Point(1, -1), Point(1, 0), Point(1, 1)
    };

    stack<Point> pointStack;
    pointStack.push(seed);

    while (!pointStack.empty()) {
        Point currPoint = pointStack.top();
        pointStack.pop();

        if (mask.at<uchar>(currPoint) == 1) continue; // 2.1. Evita punti già elaborati

        mask.at<uchar>(currPoint) = 1;
        area++;

        // 2.2. Controlla i vicini e verifica se rientrano nella soglia
        for (int i = 0; i < 8; i++) {
            Point neighborPoint = currPoint + shift8D[i];

            // 2.3. Controllo dei limiti dell'immagine
            if (neighborPoint.x < 0 || neighborPoint.x >= src.cols || neighborPoint.y < 0 || neighborPoint.y >= src.rows)
                continue;

            // 2.4. Aggiunge il pixel vicino alla regione se simile e non ancora elaborato (delta = Differenza)
            int delta = abs(src.at<uchar>(currPoint) - src.at<uchar>(neighborPoint));
            if (dst.at<uchar>(neighborPoint) == 0 && mask.at<uchar>(neighborPoint) == 0 && delta < th)
                pointStack.push(neighborPoint);
        }
    }
}

// 3. Algoritmo principale del region growing
uchar regionGrowing(const Mat &src, Mat &dst, double minSizeRegFactor, int maxRegNum, int th) {

    // 3.1. Calcola la dimensione minima della regione basata sul fattore dato
    int minSizeReg = static_cast<int>(minSizeRegFactor * src.rows * src.cols);

    // 3.2. Inizializza la matrice di output e la maschera
    uchar currLabel = 1;
    dst = Mat::zeros(src.rows, src.cols, CV_8U);
    Mat mask = Mat::zeros(src.rows, src.cols, CV_8U);

    // 3.3. Itera l'immagine pixel per pixel
    for (int x = 0; x < src.cols; x++) {
        for (int y = 0; y < src.rows; y++)
            if (dst.at<uchar>(Point(x, y)) == 0) {
                int regionArea = 0;
                grow(src, dst, mask, Point(x, y), th, regionArea);

                // 3.4. Controllo sulla dimensione della regione
                if (regionArea > minSizeReg) {
                    dst += mask * currLabel;
                    currLabel++;
                    if (currLabel > maxRegNum) {
                        cerr << "Oversegmentation." << endl;
                        return currLabel;
                    }
                } else
                    dst += mask * 255;  // 3.5. Segna come "don't care"
                mask.setTo(0);          // 3.6. Resetta la maschera
            }

        // 3.7. Mostra progresso
        if (x % (src.cols / 10) == 0)
            cout << "Progress: " << (x * 100) / src.cols << "%" << endl;
    }

    cout << "Total regions: " << static_cast<int>(currLabel) - 1 << endl;
    return currLabel;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/splash.png";
    Mat dst, src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    double minSizeRegFactor = 0.01f;
    int maxRegNum = 50;
    int th = 5;
    uchar labels = regionGrowing(src, dst, minSizeRegFactor, maxRegNum, th);
    printMasks(dst, labels);

    waitKey(0);
    return 0;
}
