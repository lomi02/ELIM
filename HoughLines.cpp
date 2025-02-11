#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void polarToCartesian(double rho, double thetaRad, Point &P1, Point &P2) {
    const int alpha = 1000;

    // 1: Punto centrale sulla retta
    double x0 = rho * cos(thetaRad);
    double y0 = rho * sin(thetaRad);

    // 2: Calcolo degli estremi della linea
    P1.x = cvRound(x0 + alpha * (-sin(thetaRad)));
    P1.y = cvRound(y0 + alpha * cos(thetaRad));
    P2.x = cvRound(x0 - alpha * (-sin(thetaRad)));
    P2.y = cvRound(y0 - alpha * cos(thetaRad));
}

void houghLines(Mat &src, Mat &dst, int CannyLTH, int CannyHTH, int HoughTH) {

    // 1: Dimensione massima dell'accumulatore
    int maxDist = hypot(src.rows, src.cols);
    vector<vector<int> > votes(2 * maxDist, vector<int>(180, 0));

    // 2: Gaussian Blur e Canny
    Mat blurred, edges;
    GaussianBlur(src, blurred, Size(3, 3), 0, 0);
    Canny(blurred, edges, CannyLTH, CannyHTH);

    // 3: Accumulatore Hough
    for (int x = 0; x < edges.rows; x++)
        for (int y = 0; y < edges.cols; y++)
            if (edges.at<uchar>(x, y) == 255)
                for (int theta = 0; theta < 180; theta++) {
                    double thetaRad = CV_PI * theta / 180; // Conversione in radianti
                    int rho = cvRound((y * cos(thetaRad)) + (x * sin(thetaRad))) + maxDist;
                    votes[rho][theta]++;
                }

    // 4: Disegno delle linee sul risultato
    dst = src.clone();
    Point P1, P2;
    for (int rho = 0; rho < votes.size(); rho++)
        for (int theta = 0; theta < votes[rho].size(); theta++)
            if (votes[rho][theta] >= HoughTH) {
                double thetaRad = CV_PI * theta / 180; // Conversione in radianti
                polarToCartesian(rho - maxDist, thetaRad, P1, P2);
                line(dst, P1, P2, Scalar(0), 2, LINE_AA);
            }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/strada.png";
    Mat dst, src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    houghLines(src, dst, 50, 150, 150);

    imshow("HoughLines", dst);
    waitKey(0);
    return 0;
}
