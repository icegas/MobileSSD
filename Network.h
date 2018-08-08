#pragma once
#define CPU_ONLY
#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;

class INetworkWorker
{
public:
	virtual ~INetworkWorker(){};
	virtual void Initialize(const string& model_file, const string& weights_file) = 0;
	virtual void Predict(const cv::Mat& img) = 0;
};

