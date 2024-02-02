#include "InferenceWithCpp.h"				

class InferenceObjectPrivate
{
public:
	InferenceObjectPrivate() :
		env(nullptr),
		session(nullptr),
		memoryInfo(nullptr),
		sessionOptions(nullptr)
	{}

	void envInit();

	std::wstring				onnxPath;
	Ort::Env					env;
	Ort::Session				session;
	Ort::MemoryInfo				memoryInfo;
	Ort::SessionOptions			sessionOptions;
	std::vector<const char*>	inputNodeNames;
	std::vector<const char*>	outputNodeNames;
	std::vector<int64_t>		inputNodeDims;
	std::vector<int64_t>		outputNodeDims;
	cv::Mat						blobMat;
};

InferenceObject::InferenceObject(const char * modelPath)
	: d_ptr(new InferenceObjectPrivate)
{
	std::string path(modelPath);
	d_ptr->onnxPath = std::wstring(path.begin(), path.end());
}

InferenceObject::~InferenceObject()
{
	delete d_ptr;
}

void InferenceObject::init()
{
	d_ptr->envInit();
}

void InferenceObject::runTest(const cv::Mat & mat)
{
	if (mat.size().empty())
	{
		return;
	}

	cv::Mat img_f32;
	mat.convertTo(img_f32, CV_32FC3);
	cv::cvtColor(img_f32, img_f32, cv::COLOR_BGR2RGB);

	// transform image format to your train model format
	// ......

	d_ptr->blobMat = cv::dnn::blobFromImage(img_f32);
	
	std::vector<Ort::Value> inputTensors;
	inputTensors.emplace_back(Ort::Value::CreateTensor<float>(d_ptr->memoryInfo,
		d_ptr->blobMat.ptr<float>(), d_ptr->blobMat.total(), d_ptr->inputNodeDims.data(),
		d_ptr->inputNodeDims.size()));
	auto outputTensors = d_ptr->session.Run(Ort::RunOptions{ nullptr }, d_ptr->inputNodeNames.data(),
		inputTensors.data(), 1, d_ptr->outputNodeNames.data(), 1);

	// 'p' pointer data is inference data
	float *p = outputTensors[0].GetTensorMutableData<float>();
}

void InferenceObjectPrivate::envInit()
{
	env = Ort::Env{ ORT_LOGGING_LEVEL_ERROR, "test" };
	bool useCUDA = true;
	if (useCUDA)
	{
		OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
		sessionOptions.SetInterOpNumThreads(1);
		sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
	}
	else
	{
		sessionOptions = Ort::SessionOptions{ nullptr };
	}

	try 
	{
		session = Ort::Session{ env, onnxPath.c_str(), sessionOptions };
	}
	catch (const Ort::Exception & e) 
	{
		printf("Error Code: [%d], Error: %s", e.GetOrtErrorCode(), e.what());
	}

	memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
	Ort::Allocator		allocator(session, memoryInfo);
	inputNodeNames.push_back(session.GetInputName(0, allocator));
	outputNodeNames.push_back(session.GetOutputName(0, allocator));
	inputNodeDims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	outputNodeDims = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
}
