#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat Otsu2k(Mat &input) {
    Mat img = input.clone();

    // 1. Computa l'istogramma
    vector<double> hist(256);
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            hist.at(img.at<uchar>(x, y))++;

    // 2. Normalizza l'istogramma
    int totalPixels = img.rows * img.cols;
    for (double &histValue: hist)
        histValue /= totalPixels;

    // 3. Calcola la media globale cumulativa
    double globCumMean = 0.0;
    for (int i = 0; i < hist.size(); i++)
        globCumMean += i * hist.at(i);

    // 4. Inizializzo probabilità, media cumulativa e varianza interclasse massima
    vector<double> prob(3);
    vector<double> cumMean(3);
    double maxVar = 0;
    vector<int> optimalTH(2);

    // 5. Computo iterativamente le soglie ottimali per massimizzare la varianza interclasse
    for (int i = 0; i < hist.size() - 2; i++) {
        prob.at(0) += hist.at(i);
        cumMean.at(0) += i * hist.at(i);

        for (int j = i + 1; j < hist.size() - 1; j++) {
            prob.at(1) += hist.at(j);
            cumMean.at(1) += j * hist.at(j);

            for (int k = j + 1; k < hist.size(); k++) {
                prob.at(2) += hist.at(k);
                cumMean.at(2) += k * hist.at(k);

                // 6. Calcolo la varianza interclasse con metodo di Otsu
                double betweenClassVariance = 0.0;
                for (int classIndex = 0; classIndex < 3; classIndex++)
                    betweenClassVariance += prob.at(classIndex) * pow(cumMean.at(classIndex) / prob.at(classIndex) - globCumMean, 2);

                // 7. Aggiorno le soglie ottimali se ho trovato una varianza più grande
                if (betweenClassVariance > maxVar) {
                    maxVar = betweenClassVariance;
                    optimalTH.at(0) = i;
                    optimalTH.at(1) = j;
                }
            }
            prob.at(2) = 0.0;
            cumMean.at(2) = 0.0;
        }
        prob.at(1) = 0.0;
        cumMean.at(1) = 0.0;
    }

    // 8. Applico il filtro gaussiano per ridurre il rumore e poi effettuo una sogliatura con le soglie ottimali
    GaussianBlur(img, img, Size(3, 3), 0, 0);
    Mat out = Mat::zeros(img.rows, img.cols, CV_8U);
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++) {
            if (img.at<uchar>(x, y) >= optimalTH.at(1))
                out.at<uchar>(x, y) = 255;
            else if (img.at<uchar>(x, y) >= optimalTH.at(0))
                out.at<uchar>(x, y) = 127;
        }


    return out;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../immagini/splash.png";
    Mat src = imread(samples::findFile(path), IMREAD_GRAYSCALE);

    //Mat src = imread(argv[1],IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    Mat dst = Otsu2k(src);
    imshow("Otsu2k", dst);
    waitKey(0);

    return 0;
}
