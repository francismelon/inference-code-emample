#pragma once
#include "../onnxruntime/onnxruntime_cxx_api.h"
#include "../opencv/opencv2/core.hpp"
#include "../opencv/opencv2/imgproc.hpp"
#include "../opencv/opencv2/highgui.hpp"
#include "../opencv/opencv2/dnn.hpp"

class InferenceObjectPrivate;
class InferenceObject
{
public:
	explicit InferenceObject(const char *modelPath);
	~InferenceObject();

	void init();

	void runTest(const cv::Mat &mat);

private:
	InferenceObjectPrivate * const d_ptr;
};