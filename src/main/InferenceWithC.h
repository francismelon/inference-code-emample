#pragma once
#include "../onnxruntime/onnxruntime_c_api.h"
#include "../opencv/opencv2/core.hpp"
#include "../opencv/opencv2/imgproc.hpp"
#include "../opencv/opencv2/highgui.hpp"
#include "../opencv/opencv2/dnn.hpp"

struct OnnxEnvObject
{
	OrtEnv					*env;
	OrtSession				*session;
	OrtMemoryInfo			*memoryInfo;
	OrtSessionOptions		*sessionOptions;
	const char				*inputNodeNames[1];
	const char				*outputNodeNames[1];
	int64_t					inputNodeDims[4];
	int64_t					outputNodeDims[4];
};

void EnvInit(OnnxEnvObject *object, const OrtApi *api);

void EnvRelease(OnnxEnvObject *object, const OrtApi *api);

void NodeInit(OnnxEnvObject *object, const OrtApi *api);

const OrtApi *GetOrtApiPtr();

void RunTest(OnnxEnvObject *object, const OrtApi * globalApi, const cv::Mat &mat);

