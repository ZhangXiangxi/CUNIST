#include "ReadData.h"

#include <cstdlib>		// import _byteswap_ulong from stdlib.h

#define bitRev(x) _byteswap_ulong(x)
#define MAGIC_NUMBER_IMAGE 2051
#define MAGIC_NUMBER_LABEL 2049

struct ImageMetaData {
	unsigned long magicNumber;
	
	unsigned long length;

	unsigned long height;
	
	unsigned long width;

	// ×ª»»´óÐ¡¶Ë
	void bitReverse(void) {
		magicNumber	= bitRev(magicNumber);
		length = bitRev(length);
		height = bitRev(height);
		width = bitRev(width);
	}
};

struct LabelMetaData {
	unsigned long magicNumber;

	unsigned long length;

	void bitReverse(void) {
		magicNumber = bitRev(magicNumber);
		length = bitRev(length);
	}
};

int readData(const char* imageFileName, const char* labelFileName, char *images, char *labels, int &width, int &height) {
	FILE* imageFile = nullptr;
	FILE* labelFile = nullptr;
	if (fopen_s(&imageFile, imageFileName, "rb") == 0) {
		puts("Can't read image file");
		goto ERROR_EXIT;
	}
	if (fopen_s(&labelFile, labelFileName, "rb") == 0) {
		puts("Can't read label file");
		goto ERROR_EXIT;
	}
	
	ImageMetaData imageMetaData;
	LabelMetaData labelMetaData;
	if (fread_s(&imageMetaData, sizeof(imageMetaData), sizeof(imageMetaData), 1, imageFile) != sizeof(imageMetaData)) {
		puts("Can't read metadata from image file");
		goto ERROR_EXIT;
	}
	if (fread_s(&labelMetaData, sizeof(labelMetaData), sizeof(labelMetaData), 1, labelFile) != sizeof(labelMetaData)) {
		puts("Can't read metadata from label file");
		goto ERROR_EXIT;
	}

	imageMetaData.bitReverse();
	labelMetaData.bitReverse();
	if (imageMetaData.magicNumber != MAGIC_NUMBER_IMAGE) {
		puts("Image magic number dismatches");
		goto ERROR_EXIT;
	}
	if (labelMetaData.magicNumber != MAGIC_NUMBER_LABEL) {
		puts("Label magic number dismatches");
		goto ERROR_EXIT;
	}
	if (imageMetaData.length != labelMetaData.length) {
		puts("Data length dismatches");
		goto ERROR_EXIT;
	}
	width = imageMetaData.width;
	height = imageMetaData.height;
	auto length = imageMetaData.length;

	if (images == nullptr || labels == nullptr) {
		fclose(imageFile);
		fclose(labelFile);
		return length;
	}

	if (fread_s(images, width * height * length * sizeof(char), sizeof(char), width * height * length, imageFile) 
		!= width * height * length * sizeof(char)) {
		puts("Error when loading data from image file");
		goto ERROR_EXIT;
	}
	if (fread_s(labels, length * sizeof(char), sizeof(char), length, labelFile) != length * sizeof(char)) {
		puts("Error when loading data from image file");
		goto ERROR_EXIT;
	}
	fclose(imageFile);
	fclose(labelFile);
	return length;

ERROR_EXIT:
	fclose(imageFile);
	fclose(labelFile);
	return 0;
}