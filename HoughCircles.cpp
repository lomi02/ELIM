#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

Mat HoughCircles(Mat &input, int HoughTH, int radMin, int radMax, int CannyLTH, int CannyHTH) {
    Mat img = input.clone();

    // 1. Inizializza l'immagine e preparala all'algoritmo
    GaussianBlur(img, img, Size(3, 3), 0);
    Canny(img, img, CannyLTH, CannyHTH);

    // 2. Inizializza l'accumulatore
    int radOffset = radMax - radMin + 1;
    int sizes[] = {img.rows, img.cols, radOffset};
    auto votes = Mat(3, sizes, CV_8U);

    // 3. Compila il diagramma dei voti
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            if (img.at<uchar>(x, y) == 255)
                for (int rad = radMin; rad < radMax; rad++)
                    for (int thetaDeg = 0; thetaDeg < 360; thetaDeg++) {
                        double thetaRad = thetaDeg * CV_PI / 180;

                        int alpha = cvRound(x - rad * cos(thetaRad));
                        int beta = cvRound(y - rad * sin(thetaRad));

                        if (alpha >= 0 && alpha < img.rows && beta >= 0 && beta < img.cols)
                            votes.at<uchar>(alpha, beta, rad - radMin)++;
                    }

    // 4. Disegna i cerchi
    Mat output = input.clone();
    for (int rad = radMin; rad < radMax; rad++)
        for (int alpha = 0; alpha < img.rows; alpha++)
            for (int beta = 0; beta < img.cols; beta++)
                if (votes.at<uchar>(alpha, beta, rad - radMin) > HoughTH)
                    circle(output, Point(beta, alpha), rad, Scalar(0), 2, 8);

    return output;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/monete.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    int HoughTH = 190;
    int radMin = 20;
    int radMax = 70;
    int CannyLTH = 40;
    int CannyHTH = 80;
    Mat dst = HoughCircles(src, HoughTH, radMin, radMax, CannyLTH, CannyHTH);

    imshow("Hough Circles", dst);
    waitKey(0);

    return 0;
}
