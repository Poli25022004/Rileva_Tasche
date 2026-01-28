#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Errore webcam\n";
        return -1;
    }

    cv::Mat frame, gray, edges;
    int saveIndex = 0;                

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::resize(frame, frame, cv::Size(640, 480));

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
        cv::Canny(gray, edges, 80, 160);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        bool foundValidContour = false;    

        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area < 2000) continue;

            std::vector<cv::Point> approx;
            cv::approxPolyDP(contours[i], approx,
                0.02 * cv::arcLength(contours[i], true), true);

            if (approx.size() < 3) continue; 

            foundValidContour = true;       

         
            cv::drawContours(frame,
                std::vector<std::vector<cv::Point>>{approx},
                -1, cv::Scalar(0, 255, 0), 2);

          
            cv::Moments m = cv::moments(approx);
            if (m.m00 != 0) {
                int cx = int(m.m10 / m.m00);
                int cy = int(m.m01 / m.m00);
                cv::putText(frame,
                    "C" + std::to_string(i),
                    cv::Point(cx, cy),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(255, 0, 0),
                    2);
            }

         
            for (size_t j = 0; j < approx.size(); j++) {
                cv::circle(frame, approx[j], 4, cv::Scalar(0, 0, 255), -1);

                cv::putText(frame,
                    std::to_string(j),
                    approx[j] + cv::Point(5, -5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 255, 255),
                    1);
            }
        }
        
        if (foundValidContour) {  
            std::string filename = "capture_" + std::to_string(saveIndex++) + ".png";
            cv::imwrite(filename, frame);
        }

        cv::imshow("Frame con indici", frame);
        cv::imshow("Edges", edges);

        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}
