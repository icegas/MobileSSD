#pragma once
//#define CPU_ONLY
//#include <caffe/caffe.hpp>
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//
//#include <algorithm>
//#include <iomanip>
//#include <iosfwd>
//#include <memory>
//#include <string>
//#include <utility>
//#include <vector>
#include "Network.h"

//using namespace caffe; // NOLINT(build/namespaces)

class Detector : INetworkWorker
{
public:
	Detector(const string& model_file, const string& weights_file);

	//Detector::~Detector();

	std::vector<vector<float> >  Detect(const cv::Mat& img);

	void Predict(const cv::Mat& img) override;
};

