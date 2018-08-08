#include "Detector.h"
#include "LayersHeader.h"
#include <boost/chrono.hpp>
#include <boost/timer/timer.hpp>

using namespace cv;
using namespace std;

#define ESC 27
#define SPACE 32 //pause button key 
#define NEXT 110 


int main(int argc, char** argv)
{
    boost::chrono::high_resolution_clock::time_point t1;
	boost::chrono::high_resolution_clock::time_point t2;

	const string model_def = argv[2];
	const string model_weights = argv[3];
	const string video = argv[1];
	
	unique_ptr<Detector> detector = make_unique<Detector>(model_def, model_weights);
	
	Mat frame;
	
	VideoCapture cap = VideoCapture(video);

	const double threshold = 0.1;

	while (true)
	{
		for (int i = 20; i; i--)
			cap >> frame;
		char key = waitKey(1);
		resize(frame, frame, Size(1000, 600));
		
		t1 = boost::chrono::high_resolution_clock::now();
		std::vector<vector<float> >  detections = detector->Detect(frame);	
		t2 = boost::chrono::high_resolution_clock::now();
		double t = (boost::chrono::duration_cast<boost::chrono::milliseconds>(t2 - t1)).count();

		for (int i = 0; i < detections.size(); i++)
		{
			const vector<float>& d = detections[i];
			float score = d[2];
			if (score > threshold)
			{
				Rect rect;
				rect.x = static_cast<int>(d[3] * frame.cols);
				rect.y = static_cast<int>(d[4] * frame.rows);
				rect.width = static_cast<int>(d[5] * frame.cols) - rect.x;
				rect.height = static_cast<int>(d[6] * frame.rows) - rect.y;
				rectangle(frame, rect, Scalar(255, 0, 0), 2, 1);

				string label = "";
				const float numLabel = static_cast<int>(d[1]);
				if (numLabel == 1)
					label = "Car";
				else
					label = "Person";
				int scoreval = floor(score * 1000);
				putText(frame, label, Point(rect.x, rect.y), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
				putText(frame, "0." + to_string(scoreval), Point(rect.x, rect.y + 15), FONT_HERSHEY_COMPLEX_SMALL, 0.6, cvScalar(0, 0, 255), 1, CV_AA);
			}

		}
		
		putText(frame, to_string(t / 1000), Point(frame.cols - 100, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(255, 0, 0), 1, CV_AA);
		
		imshow("video", frame);

		if (key == ESC)break;

		if (key == SPACE)
		{
			cout << "Pause" << endl;
			while (true)
			{
				key = waitKey(0);
				if (key == SPACE)
				{
					cout << "Play" << endl;
					break;
				}
			}
		}

		if (key == NEXT)
		{
			cout << "Enter number of seconds" << endl;
			double seconds;
			cin >> seconds;
			seconds *= 25;

			for (int i = seconds; i; i--)
			{
				cap >> frame;
			}
		}
	}

	
	
}