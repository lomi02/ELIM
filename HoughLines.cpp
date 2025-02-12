#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat HoughLines(Mat &input, int HoughTH, int CannyLTH, int CannyHTH) {

    // 1. Inizializza l'immagine e preparala all'algoritmo
    Mat img = input.clone();
    GaussianBlur(img, img, Size(1, 1), 0);
    Canny(img, img, CannyLTH, CannyHTH);

    // 2. Inizializza l'accumulatore
    int diagLen = hypot(img.rows, img.cols);
    int maxTheta = 180;
    Mat votes = Mat::zeros(diagLen * 2, maxTheta, CV_8U);

    // 3. Compila il diagramma dei voti
    for (int x = 0; x < img.rows; x++) {
        for (int y = 0; y < img.cols; y++) {
            if (img.at<uchar>(x, y) == 255) {
                for (int theta = 0; theta < maxTheta; theta++) {
                    int rho = cvRound(y * cos(theta) + x * sin(theta));
                    int rhoIndex = rho + diagLen;
                    votes.at<uchar>(rhoIndex, theta)++;
                }
            }
        }
    }

    // 4. Disegna le linee
    int alpha = diagLen * 2;
    Mat output = input.clone();
    for (int rhoIndex = 0; rhoIndex < votes.rows; rhoIndex++) {
        for (int theta = 0; theta < votes.cols; theta++) {
            if (votes.at<uchar>(rhoIndex, theta) > HoughTH) {
                int rho = rhoIndex - diagLen;

                int x0 = cvRound(rho * cos(theta));
                int y0 = cvRound(rho * sin(theta));

                Point p1;
                p1.x = cvRound(x0 + alpha * -sin(theta));
                p1.y = cvRound(y0 + alpha * cos(theta));

                Point p2;
                p2.x = cvRound(x0 - alpha * -sin(theta));
                p2.y = cvRound(y0 - alpha * cos(theta));

                line(output, p1, p2, Scalar(0), 2, 0);
            }
        }
    }
    return output;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/strada.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    int HoughTH = 150;
    int CannyLTH = 40;
    int CannyHTH = 80;
    Mat dst = HoughLines(src, HoughTH, CannyLTH, CannyHTH);

    imshow("HoughLines", dst);
    waitKey(0);
    return 0;
}
