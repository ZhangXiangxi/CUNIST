#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "ReadData.h"

using namespace std;

const char* trainImageFileName = "train-images.idx3-ubyte";
const char* trainLabelFileName = "train-labels.idx1-ubyte";
const char* testImageFileName = "t10k-images.idx3-ubyte";
const char* testLabelFileName = "t10k-labels.idx1-ubyte";

int main() {
	char* images;
	char* labels;
	int trainLength;
	int width;
	int height;
	trainLength = readData(trainImageFileName, trainLabelFileName, nullptr, nullptr, width, height);
	cout << "The train data total is: " << trainLength << endl;
	cout << "The image width is: " << width << endl;
	cout << "The image height is: " << height << endl;
	system("pause");
	return 0;
}


