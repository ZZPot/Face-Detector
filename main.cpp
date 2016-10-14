#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <string>
#include <iostream>

using namespace cv;

std::vector<std::string> cascade_files = {	"cascadeFiles/haarcascade_frontalface_alt.xml",
											"cascadeFiles/haarcascade_eye.xml"};
#define MASK_IMG		"mask.png"
#define GLASSES_PNG		"glasses.png"
#define WND_NAME_RES	"Result"

Scalar bg_color(255, 0, 255);

void OverlayImg(Mat img, Mat mask, Rect place, float opacity = 1.0f);

int main(int argc, char* argv[])
{
    CascadeClassifier face_cascade, eye_cascade;
    face_cascade.load(cascade_files[0]);
	eye_cascade.load(cascade_files[1]);
    Mat mask_img = imread(MASK_IMG);
    Mat glasses_img = imread(GLASSES_PNG);

    Mat frame, frame_gray;

    VideoCapture cap(0);
    if(!cap.isOpened())
	{
		std::cout << "Need webcam as input\n";
		return -1;
	}

    namedWindow(WND_NAME_RES);
    std::vector<Rect> faces;
    bool overlay_mask = false;
	bool overlay_glasses = true;
    while(true)
    {
        // Capture the current frame
        cap >> frame;
        cvtColor(frame, frame_gray, CV_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);
        
        face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));
        
        for(int i = 0; i < faces.size(); i++)
        {
			if(overlay_mask)
				OverlayImg(frame, mask_img, faces[i]);
			if(overlay_glasses)
			{
				Mat faceROI = frame_gray(faces[i]);
				std::vector<Rect> eyes;
				eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30));
            
				if(eyes.size() >= 2)
				{
					Rect eyes_place(min(eyes[0].x, eyes[1].x), min(eyes[0].y, eyes[1].y),
									abs(eyes[0].x - eyes[1].x) + (eyes[0].width + eyes[1].width) / 2,
									abs(eyes[0].y - eyes[1].y) + (eyes[0].height + eyes[1].height) / 2); // average size of both eyes
					eyes_place.x -= (eyes[0].width + eyes[1].width) / 5 - faces[i].x;
					eyes_place.width += 2*(eyes[0].width + eyes[1].width)/5;
					eyes_place.y += faces[i].y  + 10;
					eyes_place.height -= 10;
					//eyes_place.height += (eyes[0].height + eyes[1].height) * 0.66;

					OverlayImg(frame, glasses_img, eyes_place, 0.7);
				}
			}
        }
        imshow(WND_NAME_RES, frame);
        if (waitKey(33) == 27)
            break;
    }
    cap.release();
    destroyAllWindows();
    return 0;
}

void OverlayImg(Mat img, Mat mask, Rect place, float opacity)
{            
	Mat resized_mask;
	resize(mask, resized_mask, Size(place.width, place.height));
	Mat mask_mask; // lol
	inRange(resized_mask, bg_color, bg_color, mask_mask);
	mask_mask = 255 - mask_mask;
	erode(mask_mask, mask_mask, getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3 )));
	Mat imgROI = img(place);
	add(resized_mask * opacity, imgROI * (1.0 - opacity), resized_mask, mask_mask);
	resized_mask.copyTo(imgROI, mask_mask);
}