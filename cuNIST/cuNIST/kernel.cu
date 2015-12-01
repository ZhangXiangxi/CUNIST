#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "ReadData.h"

using namespace std;

const char* trainImageFileName = "train-images.idx3-ubyte";
const char* trainLabelFileName = "train-labels.idx1-ubyte";
const char* testImageFileName = "t10k-images.idx3-ubyte";
const char* testLabelFileName = "t10k-labels.idx1-ubyte";

class SingleImage {
public:
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
	void print() const {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				printf("%d\t", 0 + data[i * width + j]);
			}
			putchar('\n');
		}
	}
};

int previousTest(void) {
	unsigned char* images;
	unsigned char* labels;
	unsigned char** multiImages;  //多个图片的数组
	int trainLength;
	int width;
	int height;
	trainLength = readData(trainImageFileName, trainLabelFileName, nullptr, nullptr, width, height);
	cout << "The train data total is: " << trainLength << endl;
	cout << "The image width is: " << width << endl;
	cout << "The image height is: " << height << endl;
	images = new unsigned char[trainLength * width * height];
	labels = new unsigned char[trainLength];
	multiImages = new unsigned char *[trainLength];
	for (auto i = 0; i != trainLength; i++) {
		multiImages[i] = new unsigned char[width * height];
	}

	if (readData(trainImageFileName, trainLabelFileName, images, labels, width, height) != trainLength)
		return 1;
	changeImageArray(images, multiImages, width, height, trainLength);
	char bmpfile[] = "image.bmp";
	toBMPImage(bmpfile, images, width, height);
	printf("The label is %d\n", labels[0]);
	SingleImage singleImage(width, height, images);
	singleImage.print();
	delete[] images;
	delete[] labels;
	for (auto i = 0; i != trainLength; i++) {
		delete[] multiImages[i];
	}
	delete multiImages;
	system("pause");
	return 0;
}
