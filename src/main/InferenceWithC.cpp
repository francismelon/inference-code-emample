#include <stdio.h>
#include "InferenceWithC.h"

#define OBJECT_STR(OBJECT) #OBJECT
#define CREATE_OBJECT_FAILED(OBJECT)	\
	printf("Creat object %s failed!\n", OBJECT_STR(OBJECT))

void EnvInit(OnnxEnvObject * object, const OrtApi *api)
{
	if (!api || !object)
	{
		return;
	}

	(*object).env					= NULL;
	(*object).session				= NULL;
	(*object).memoryInfo			= NULL;
	(*object).sessionOptions		= NULL;

	api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "test", &(object->env));
	if (!object->env)
	{
		CREATE_OBJECT_FAILED(envObject.env);
		return;
	}

	api->CreateSessionOptions(&(object->sessionOptions));
	if (!object->sessionOptions)
	{
		CREATE_OBJECT_FAILED(object->sessionOptions);
		return;
	}

	bool useCUDA = true;
	if (useCUDA)
	{
		OrtSessionOptionsAppendExecutionProvider_CUDA(object->sessionOptions, 0);
		api->SetInterOpNumThreads(object->sessionOptions, 1);
		api->SetSessionGraphOptimizationLevel(object->sessionOptions, ORT_ENABLE_ALL);
	}

	const ORTCHAR_T *modelPath = L"test.onnx";
	api->CreateSession(object->env, modelPath, object->sessionOptions, &(object->session));
	if (!object->session)
	{
		CREATE_OBJECT_FAILED(envObject.session);
		return;
	}

	api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &(object->memoryInfo));
	if (!object->memoryInfo)
	{
		CREATE_OBJECT_FAILED(object->memoryInfo);
		return;
	}
}

void EnvRelease(OnnxEnvObject * object, const OrtApi *api)
{
	if (!api || !object)
	{
		return;
	}

	api->ReleaseMemoryInfo((*object).memoryInfo);
	api->ReleaseSessionOptions((*object).sessionOptions);
	api->ReleaseSession((*object).session);
	api->ReleaseEnv((*object).env);
}

void NodeInit(OnnxEnvObject * object, const OrtApi *api)
{
	if (!object)
	{
		return;
	}

	OrtAllocator		*inputAllocator = NULL;
	char				*inputNodeName	= {};
	api->CreateAllocator(object->session, object->memoryInfo, &inputAllocator);
	api->SessionGetInputName(object->session, 0, inputAllocator, &inputNodeName);
	(*object).inputNodeNames[0] = inputNodeName;
	api->ReleaseAllocator(inputAllocator);

	OrtAllocator		*outputAllocator = NULL;
	char				*outputNodeName  = {};
	api->CreateAllocator(object->session, object->memoryInfo, &outputAllocator);
	api->SessionGetOutputName(object->session, 0, outputAllocator, &outputNodeName);
	(*object).outputNodeNames[0] = outputNodeName;
	api->ReleaseAllocator(outputAllocator);

	size_t				dimCount		= 0;
	OrtTypeInfo			*typeInfo		= NULL;
	OrtTensorTypeAndShapeInfo *tensorInfo = NULL;
	api->SessionGetInputTypeInfo(object->session, 0, &typeInfo);
	api->CastTypeInfoToTensorInfo(typeInfo, (const OrtTensorTypeAndShapeInfo**)&tensorInfo);
	api->GetDimensionsCount(tensorInfo, &dimCount);
	api->GetDimensions(tensorInfo, object->inputNodeDims, dimCount);
	api->ReleaseTypeInfo(typeInfo);

	api->SessionGetOutputTypeInfo(object->session, 0, &typeInfo);
	api->CastTypeInfoToTensorInfo(typeInfo, (const OrtTensorTypeAndShapeInfo**)&tensorInfo);
	api->GetDimensionsCount(tensorInfo, &dimCount);
	api->GetDimensions(tensorInfo, object->outputNodeDims, dimCount);
	api->ReleaseTypeInfo(typeInfo);
}

const OrtApi * GetOrtApiPtr()
{
	const OrtApiBase *baseApi = OrtGetApiBase();
	if (baseApi)
	{
		return baseApi->GetApi(ORT_API_VERSION);
	}

	return NULL;
}

void RunTest(OnnxEnvObject *object, const OrtApi * globalApi, const cv::Mat &mat)
{
	if (!globalApi || !object)
	{
		if (!globalApi)
		{
			printf("Get OrtApi failed, please update latest release package!\n");
		}		
		return;
	}

	cv::Mat img_f32;
	mat.convertTo(img_f32, CV_32FC3);
	cv::cvtColor(img_f32, img_f32, cv::COLOR_BGR2RGB);

	// transform image format to your train model format
	// ......

	cv::Mat blobMat = cv::dnn::blobFromImage(img_f32);

	OrtValue *inputTensor = NULL;
	size_t inputSize = sizeof(object->inputNodeDims) / sizeof(object->inputNodeDims[0]);
	size_t outputSize = sizeof(object->outputNodeDims) / sizeof(object->outputNodeDims[0]);
	globalApi->CreateTensorWithDataAsOrtValue(object->memoryInfo, blobMat.ptr<float>(),
		blobMat.total() * sizeof(float), object->inputNodeDims, inputSize,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputTensor);

	OrtValue *outputTensor = NULL;
	globalApi->Run(object->session, NULL, object->inputNodeNames, (const OrtValue* const*)&inputTensor,
		1, object->outputNodeNames, 1, &outputTensor);

	// 'p' pointer data is inference data
	float *p = NULL;
	globalApi->GetTensorMutableData(outputTensor, (void**)&p);
	printf("batch:[%lld], channel:[%lld], height:[%lld], weight:[%lld]\n", object->outputNodeDims[0],
		object->outputNodeDims[1], object->outputNodeDims[2], object->outputNodeDims[3]);
	
}
