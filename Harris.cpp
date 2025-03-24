#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat harris(Mat &input, float k, int threshTH) {
    Mat img = input.clone();

    // 1. Computo la derivata orizzontale e verticale
    Mat Dx, Dy;
    Sobel(img, Dx, CV_32F, 1, 0, 3);
    Sobel(img, Dy, CV_32F, 0, 1, 3);

    // 2. Calcolo il prodotto delle derivate parziali
    Mat DxDy;
    multiply(Dx, Dy, DxDy);

    // 3. Elevo le derivate in potenza di 2 e uso il filtro gaussiano
    Mat Dx2, Dy2;
    pow(Dx, 2, Dx2);
    pow(Dy, 2, Dy2);
    GaussianBlur(Dx2, Dx2, Size(3, 3), 0, 0);
    GaussianBlur(Dy2, Dy2, Size(3, 3), 0, 0);
    GaussianBlur(DxDy, DxDy, Size(3, 3), 0, 0);

    // 4. Calcolo le diagonali della matrice di derivazione
    Mat mainDiagMult, secondDiagMult;
    multiply(Dx2, Dy2, mainDiagMult);
    multiply(DxDy, DxDy, secondDiagMult);

    // 5. Computo i componenti per la formula di Harris
    Mat det = mainDiagMult - secondDiagMult, trace;
    pow(Dx2 + Dy2, 2, trace);

    // 6. Formula di Harris
    Mat R = det - (k * trace);

    // 7. Applico la normalizzazione e la sogliatura per individuare i bordi
    normalize(R, R, 0, 255, NORM_MINMAX, CV_8U);
    threshold(R, R, threshTH, 255, THRESH_BINARY);

    // 8. Cerchio i corner dell'immagine
    Mat out = input.clone();
    for (int x = 0; x < R.rows; x++)
        for (int y = 0; y < R.cols; y++)
            if (R.at<uchar>(x, y) > 0)
                circle(out, Point(y, x), 3, Scalar(0), 1, 8, 0);

    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/foglia.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    float k = 0.05;
    int threshTH = 145;
    Mat dst = harris(src, k, threshTH);

    imshow("Harris", dst);
    waitKey(0);

    return 0;
}
