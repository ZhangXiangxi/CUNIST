#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "ReadData.h"

using namespace std;

const char* trainImageFileName = "train-images.idx3-ubyte";
const char* trainLabelFileName = "train-labels.idx1-ubyte";
const char* testImageFileName = "t10k-images.idx3-ubyte";
const char* testLabelFileName = "t10k-labels.idx1-ubyte";

struct SingleImage {
	const int width;
	const int height;
	unsigned char * data;
	SingleImage(int width, int height, unsigned char* sourceData) : width(width), height(height) {
		data = new unsigned char[width * height];
		memcpy_s(data, width*height, sourceData, width*height);
	}
	~SingleImage() {
		delete[] data;
	}
};

void printSingleImage(SingleImage image) {
	int temp;
	for (int i = 0; i < image.height; i++) {
		for (int j = 0; j < image.width; j++) {
			printf("%d\t", image.data[i * image.width + j]);
		}
		putchar('\n');
	}
}

int main() {
	unsigned char* images;
	unsigned char* labels;
	int trainLength;
	int width;
	int height;
	trainLength = readData(trainImageFileName, trainLabelFileName, nullptr, nullptr, width, height);
	cout << "The train data total is: " << trainLength << endl;
	cout << "The image width is: " << width << endl;
	cout << "The image height is: " << height << endl;
	images = new unsigned char[trainLength * width * height];
	labels = new unsigned char[trainLength];
	

	if (readData(trainImageFileName, trainLabelFileName, images, labels, width, height) != trainLength)
		return 1;
	SingleImage singleImage(width, height, images);
	printSingleImage(singleImage);
	delete[] images;
	delete[] labels;
	system("pause");
	return 0;
}


