#include "ReadData.h"

#include <cstdlib>		// import _byteswap_ulong from stdlib.h
#include <cstdio>

#define bitRev(x) _byteswap_ulong(x)
#define MAGIC_NUMBER_IMAGE 2051
#define MAGIC_NUMBER_LABEL 2049

struct ImageMetaData {
	unsigned long magicNumber;		// У����
	
	unsigned long length;			// ������Ŀ
	
	unsigned long height;			// ͼƬ�߶�
	
	unsigned long width;			// ͼƬ���

	// ת����С��
	void bitReverse(void) {
		magicNumber	= bitRev(magicNumber);
		length = bitRev(length);
		height = bitRev(height);
		width = bitRev(width);
	}
};

struct LabelMetaData {
	unsigned long magicNumber;		// У����

	unsigned long length;			// ������Ŀ

	void bitReverse(void) {
		magicNumber = bitRev(magicNumber);
		length = bitRev(length);
	}
};

int readData(const char* imageFileName, const char* labelFileName, unsigned char *images, unsigned char *labels, int &width, int &height) {
	FILE *imageFile;
	FILE *labelFile;

	// ���ļ�
	if (fopen_s(&imageFile, imageFileName, "rb") != 0) {
		puts("Can't read image file");
		return 0;
	}
	if (fopen_s(&labelFile, labelFileName, "rb") != 0) {
		puts("Can't read label file");
		fclose(imageFile);
		return 0;
	}
	
	// �����ļ�ͷ(Ԫ����)
	ImageMetaData imageMetaData;
	LabelMetaData labelMetaData;
	if (fread_s(&imageMetaData, sizeof(imageMetaData), sizeof(imageMetaData), 1, imageFile) != 1) {
		puts("Can't read metadata from image file");
		goto ERROR_EXIT;
	}
	if (fread_s(&labelMetaData, sizeof(labelMetaData), sizeof(labelMetaData), 1, labelFile) != 1) {
		puts("Can't read metadata from label file");
		goto ERROR_EXIT;
	}

	// ת����С��
	imageMetaData.bitReverse();
	labelMetaData.bitReverse();

	// ����������Ч��
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

	// ��ȡ���������
	width = imageMetaData.width;
	height = imageMetaData.height;
	auto length = imageMetaData.length;

	// �������ָ���п�ָ�룬�Ͳ���ȡ
	if (images == nullptr || labels == nullptr) {
		fclose(imageFile);
		fclose(labelFile);
		return length;
	}

	// ��ȡ����
	if (fread_s(images, width * height * length * sizeof(unsigned char), sizeof(unsigned char), width * height * length, imageFile) != width * height * length) {
		puts("Error when loading data from image file");
		goto ERROR_EXIT;
	}
	if (fread_s(labels, length * sizeof(unsigned char), sizeof(unsigned char), length, labelFile) != length) {
		puts("Error when loading data from image file");
		goto ERROR_EXIT;
	}

	// �ر��ļ�
	fclose(imageFile);
	fclose(labelFile);
	return length;

ERROR_EXIT:					// �쳣����
	fclose(imageFile);
	fclose(labelFile);
	return 0;
}

void changeImageArray(const unsigned char *images, unsigned char **destImages, const unsigned long width, const unsigned long height, const unsigned long length){
	for (auto i = 0; i != length; i++) {
		for (auto j = 0; j != width * height; j++) {
			destImages[i][j] = images[i*width*height + j];
		}
	}
	return;
}

void toBMPImage(const char* imageFileName, const unsigned char *images, const unsigned long width, const unsigned long height, const unsigned long length){

}