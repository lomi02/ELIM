#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

/**
 * Applica l'algoritmo di trasformata di Hough per rilevare cerchi in un'immagine in input.
 *
 * Questa funzione esegue i seguenti passaggi:
 * 1. Applica un filtro gaussiano per ridurre il rumore
 * 2. Esegue il rilevamento dei bordi tramite l'algoritmo di Canny
 * 3. Popola un accumulatore 3D (x, y, raggio) votando per i possibili centri dei cerchi
 * 4. Seleziona i punti con voti sufficienti e disegna i cerchi corrispondenti sull'immagine originale
 *
 * @param input     Immagine in input (scala di grigi)
 * @param houghTH   Soglia minima di voti nell'accumulatore per considerare un cerchio valido
 * @param Rmin      Raggio minimo dei cerchi da rilevare
 * @param Rmax      Raggio massimo dei cerchi da rilevare
 *
 * @return Immagine con i cerchi rilevati disegnati (cerchi neri)
 */
Mat hough_circles(Mat &input, int houghTH, int Rmin, int Rmax) {
    Mat img = input.clone();

    // Passo 1: Riduzione del rumore tramite filtro gaussiano 5x5
    GaussianBlur(img, img, Size(5, 5), 0.5, 0.5);

    // Passo 2: Rilevamento dei bordi con Canny
    Canny(img, img, 100, 250);

    // Passo 3: Creazione dell'accumulatore 3D (rows x cols x raggio)
    vector<vector<vector<int> > > votes(img.rows, vector<vector<int> >(img.cols, vector<int>(Rmax - Rmin + 1, 0)));

    // Ciclo su tutti i pixel dell'immagine dei bordi
    for (int x = 0; x < img.rows; x++)
        for (int y = 0; y < img.cols; y++)
            if (img.at<uchar>(x, y) == 255)         // Solo pixel di bordo
                for (int r = Rmin; r < Rmax; r++)           // Ciclo su tutti i raggi possibili
                    for (int theta = 0; theta < 360; theta++) {     // Ciclo sugli angoli

                        // Calcola le coordinate del possibile centro del cerchio
                        int a = y - static_cast<int>(r * cos(theta * CV_PI / 180.0));
                        int b = x - static_cast<int>(r * sin(theta * CV_PI / 180.0));

                        // Controlla che le coordinate siano valide nell'immagine
                        if (a >= 0 && a < img.cols && b >= 0 && b < img.rows)
                            votes[b][a][r - Rmin]++; // Incrementa il voto nell'accumulatore
                    }

    // Passo 4: Disegno dei cerchi con voti sufficienti sull'immagine originale
    Mat out = input.clone();
    for (int a = 0; a < img.cols; a++)
        for (int b = 0; b < img.rows; b++)
            for (int r = Rmin; r < Rmax; r++)
                if (votes[b][a][r - Rmin] > houghTH)
                    circle(out, Point(a, b), r, Scalar(0), 1); // Cerchio nero di spessore 1

    return out;
}
