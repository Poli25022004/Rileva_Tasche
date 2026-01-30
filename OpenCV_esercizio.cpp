#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Errore webcam\n";
        return -1;
    }

    Mat frame, gray, edges;
    int saveIndex = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        resize(frame, frame, Size(640, 480));

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(5, 5), 0);
        Canny(gray, edges, 80, 160);

        vector<vector<Point>> contours;
        findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        bool foundValidContour = false;

        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area < 2000) continue;

            vector<Point> approx;
            approxPolyDP(
                contours[i],
                approx,
                0.02 * arcLength(contours[i], true),
                true
            );

            if (approx.size() < 3) continue;

            foundValidContour = true;

            drawContours(
                frame,
                vector<vector<Point>>{approx},
                -1,
                Scalar(0, 255, 0),
                2
            );

            Moments m = moments(approx);
            if (m.m00 != 0) {
                int cx = int(m.m10 / m.m00);
                int cy = int(m.m01 / m.m00);
                putText(
                    frame,
                    "C" + to_string(i),
                    Point(cx, cy),
                    FONT_HERSHEY_SIMPLEX,
                    0.6,
                    Scalar(255, 0, 0),
                    2
                );
            }

            for (size_t j = 0; j < approx.size(); j++) {
                circle(frame, approx[j], 4, Scalar(0, 0, 255), -1);
                putText(
                    frame,
                    to_string(j),
                    approx[j] + Point(5, -5),
                    FONT_HERSHEY_SIMPLEX,
                    0.5,
                    Scalar(0, 255, 255),
                    1
                );
            }
        }

        if (foundValidContour) {
            string filename = "capture_" + to_string(saveIndex++) + ".png";
            imwrite(filename, frame);
        }

        imshow("Frame con indici", frame);
        imshow("Edges", edges);

        if (waitKey(1) == 27) break;
    }

    return 0;
}
