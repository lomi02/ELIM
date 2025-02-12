#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat canny(Mat &input, int CannyLTH, int CannyHTH) {
    Mat img = input.clone();

    // 1. Applico il filtro gaussiano
    GaussianBlur(img, img, Size(3, 3), 0, 0);

    // 2. Computo il gradiente, la magnitudo e gli orientamenti
    Mat Dx, Dy, orientations;
    Sobel(img, Dx, CV_32F, 1, 0);
    Sobel(img, Dy, CV_32F, 0, 1);
    Mat mag = abs(Dx) + abs(Dy);
    normalize(mag, mag, 0, 255, NORM_MINMAX, CV_8U);
    phase(Dx, Dy, orientations, true);

    // 3. Eseguo il Non-Maximum Suppression
    Mat nms = mag.clone();
    uchar px1, px2;
    for (int i = 1; i < mag.rows - 1; i++) {
        for (int j = 1; j < mag.cols - 1; j++) {
            float angle = orientations.at<float>(i, j);

            // 0°(+-22.5) && 180°(+-22.5)
            if ((angle >= 0 && angle <= 22.5) || (angle >= 157.5 && angle <= 180)) {
                px1 = mag.at<uchar>(i, j + 1);
                px2 = mag.at<uchar>(i, j - 1);
            }
            // 45°(+-22.5)
            else if (angle > 22.5 && angle <= 67.5) {
                px1 = mag.at<uchar>(i + 1, j + 1);
                px2 = mag.at<uchar>(i - 1, j - 1);
            }
            // 90°(+-22.5)
            else if (angle > 67.5 && angle <= 112.5) {
                px1 = mag.at<uchar>(i + 1, j);
                px2 = mag.at<uchar>(i - 1, j);
            }
            // 135°(+-22.5)
            else {
                px1 = mag.at<uchar>(i + 1, j - 1);
                px2 = mag.at<uchar>(i - 1, j + 1);
            }

            uchar curMag = mag.at<uchar>(i, j);
            if (curMag < px1 || curMag < px2)
                nms.at<uchar>(i, j) = 0;
        }
    }

    // 4. Eseguo l'hysteresis thresholding
    Mat out = Mat::zeros(nms.rows, nms.cols, nms.type());
    for (int i = 0; i < nms.rows; i++) {
        for (int j = 0; j < nms.cols; j++) {
            if (nms.at<uchar>(i, j) > CannyHTH) {
                out.at<uchar>(i, j) = 255;

                // 5. Definisco una regione d'interesse (ROI) per un thresholding locale
                Rect roi(i - 1, j - 1, 3, 3);
                for (int k = roi.x; k < roi.x + roi.height; k++) {
                    for (int l = roi.y; l < roi.y + roi.width; l++) {
                        if (nms.at<uchar>(k, l) > CannyLTH and nms.at<uchar>(k, l) < CannyHTH)
                            out.at<uchar>(k, l) = 255;
                    }
                }
            }
        }
    }
    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/fiore.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    int CannyTHL = 5;
    int CannyTHH = 20;

    Mat dst = canny(src, CannyTHL, CannyTHH);

    imshow("Canny", dst);
    waitKey(0);
    return 0;
}
