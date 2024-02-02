#include <iostream>
#include "InferenceWithC.h"
#include "InferenceWithCpp.h"

void test_c()
{
	OnnxEnvObject object;
	const OrtApi *api = GetOrtApiPtr();
	EnvInit(&object, api);
	NodeInit(&object, api);
	cv::Mat mat = cv::imread("./1.jpg");
	RunTest(&object, api, mat);

	EnvRelease(&object, api);
}

void test_cpp()
{
	const char *path = "test.onnx";
	InferenceObject object(path);
	object.init();
	cv::Mat mat = cv::imread("./1.jpg");
	object.runTest(mat);
}

int main()
{
    std::cout << "Hello World!\n";
	
	std::cout << "c test begin:\n";
	test_c();

	std::cout << "\ncpp test begin:\n";
	test_cpp();
}
