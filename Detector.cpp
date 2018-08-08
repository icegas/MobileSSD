#include "Detector.h"

Detector::Detector(const string& model_file, const string& weights_file)
{
	this->Initialize(model_file, weights_file);	
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) 
{
	this->Predict(img);
	Blob<float>* result_blob = net_->output_blobs()[0];
	const float* result = result_blob->cpu_data();
	const int num_det = result_blob->height();
	vector<vector<float> > detections;
	for (int k = 0; k < num_det; ++k) {
		if (result[0] == -1) {
			// Skip invalid detection.
			result += 7;
			continue;
		}
		vector<float> detection(result, result + 7);
		detections.push_back(detection);
		result += 7;
	}
	return detections;	
}

void Detector::Predict(const cv::Mat& img)
{
	INetworkWorker::Predict(img);
}