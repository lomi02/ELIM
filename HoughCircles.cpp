#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void houghCircles(Mat &src, Mat &dst, int CannyLTH, int CannyHTH, int HoughTH, int Rmin, int Rmax) {
    dst = src.clone();

    // 1: Precompila i valori di seno e coseno
    vector<double> cos_theta(360);
    vector<double> sin_theta(360);
    for (int theta = 0; theta < 360; theta++) {
        cos_theta[theta] = cos(theta * CV_PI / 180);
        sin_theta[theta] = sin(theta * CV_PI / 180);
    }

    // 2: Inizializzazione della matrice di voti
    vector<vector<vector<int> > > votes(src.rows, vector<vector<int> >(src.cols, vector<int>(Rmax - Rmin + 1, 0)));

    // 3: Gaussian Blur e Canny
    Mat blurred, edges;
    GaussianBlur(src, blurred, Size(7, 7), 0, 0);
    Canny(blurred, edges, CannyLTH, CannyHTH);

    // 4: Accumulo dei voti per i possibili cerchi
    for (int x = 0; x < edges.rows; x++)
        for (int y = 0; y < edges.cols; y++)
            if (edges.at<uchar>(x, y) == 255)
                for (int r = Rmin; r <= Rmax; r++)
                    for (int theta = 0; theta < 360; theta++) {
                        int a = y - r * cos_theta[theta];
                        int b = x - r * sin_theta[theta];
                        if (a >= 0 && a < edges.cols && b >= 0 && b < edges.rows)
                            votes[b][a][r - Rmin]++;
                    }

    // 5: Disegno dei cerchi rilevati sull'immagine di destinazione
    for (int r = Rmin; r <= Rmax; r++)
        for (int b = 0; b < edges.rows; b++)
            for (int a = 0; a < edges.cols; a++)
                if (votes[b][a][r - Rmin] > HoughTH) {
                    circle(dst, Point(a, b), 3, Scalar(0), 2, 8, 0);
                    circle(dst, Point(a, b), r, Scalar(0), 2, 8, 0);
                }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/monete.png";
    Mat dst, src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    houghCircles(src, dst, 100, 200, 150, 20, 100);

    imshow("Hough Circles", dst);
    waitKey(0);

    return 0;
}
