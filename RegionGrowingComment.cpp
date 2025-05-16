#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Controlla se un dato punto è all'interno dei limiti dell'immagine.
 *
 * @param img   Immagine di input.
 * @param neigh Punto da verificare.
 *
 * @return True se il punto è dentro i limiti dell'immagine, false altrimenti.
 */
bool inRange(Mat &img, Point neigh) {
    return neigh.x >= 0 and neigh.x < img.cols and neigh.y >= 0 and neigh.y < img.rows;
}

/**
 * Controlla se l'intensità di un dato pixel è simile a quella del pixel seed.
 *
 * @param img     Immagine di input.
 * @param seed    Punto seed.
 * @param neigh   Punto vicino da verificare.
 * @param similTH Soglia di similarità dell'intensità.
 *
 * @return True se la differenza di intensità è entro la soglia, false altrimenti.
 */
bool isSimilar(Mat &img, Point seed, Point neigh, int similTH) {
    int seedIntensity = img.at<uchar>(seed);
    int currIntensity = img.at<uchar>(neigh);
    int intensityDelta = std::abs(seedIntensity - currIntensity);
    return intensityDelta < similTH;
}

/**
 * Esegue la segmentazione tramite crescita di regione su un'immagine in scala di grigi a partire da un punto seed.
 *
 * La crescita di regione è una tecnica di segmentazione basata su regioni che raggruppa
 * i pixel vicini con valori di intensità simili in una stessa regione.
 *
 * @param input    Immagine di input in scala di grigi.
 * @param similTH  Soglia di similarità di intensità per la crescita della regione.
 * @param seed     Punto seed (default è l'angolo in alto a sinistra, cv::Point(0, 0)).
 *
 * @return Immagine binaria che evidenzia la regione segmentata.
 */
Mat region_growing(Mat &input, int similTH, Point seed = Point(0, 0)) {
    Mat img = input.clone();

    // Crea un'immagine di output della stessa dimensione, inizializzata a nero.
    Mat segmentedImg = Mat::zeros(img.size(), CV_8U);

    // Inizializza una coda per l'attraversamento dei pixel, partendo dal punto seed.
    std::queue<Point> pixelQueue;
    pixelQueue.push(seed);

    // Esegui la crescita della regione usando un approccio breadth-first search (BFS).
    while (not pixelQueue.empty()) {
        Point currentPixel = pixelQueue.front();
        pixelQueue.pop();

        // Controlla se il pixel corrente non è stato ancora visitato.
        if (segmentedImg.at<uchar>(currentPixel) == 0) {
            segmentedImg.at<uchar>(currentPixel) = 255; // Segna il pixel come parte della regione segmentata.

            // Definisce una regione di interesse 3x3 attorno al pixel corrente.
            Rect regionOfInterest(currentPixel.x - 1, currentPixel.y - 1, 3, 3);

            // Itera attraverso i pixel vicini nella regione di interesse.
            for (int roi_y = regionOfInterest.y; roi_y < regionOfInterest.y + regionOfInterest.height; ++roi_y)
                for (int roi_x = regionOfInterest.x; roi_x < regionOfInterest.x + regionOfInterest.width; ++roi_x) {
                    Point neighborPixel(roi_x, roi_y);

                    // Controlla se il pixel vicino è dentro i limiti dell'immagine ed ha un'intensità simile al pixel seed.
                    if (inRange(img, neighborPixel) and isSimilar(img, seed, neighborPixel, similTH))

                        // Aggiunge il pixel vicino alla coda.
                        pixelQueue.push(neighborPixel);
                }
        }
    }

    return segmentedImg;
}
