#ifndef __CUNIST_READDATA_H
#define __CUNIST_READDATA_H

/**
 *这个函数从指定的文件名中读入数据到指定指针指向的内容中，解析了 MNIST 的数据格式
 *如果数据指针为空指针，则只返回长度，不读取数据
 *如果发生错误，返回0
 *
 *@param imageFileName: 图片文件的文件名
 *@param labelFileName: 标注文件的文件名
 *@param images: 指向数据集所用空间的指针
 *@param labels: 指向标注集所用空间的指针
 *@param width: 图像的宽度
 *@param height: 图像的高度
 *@return: 返回数据的数目
 *
*/

int readData(const char* imageFileName, const char* labelFileName, unsigned char *images, unsigned char *labels, int &width, int &height);
/**
* 将图片数组转换为多个图片的二维数组
**/
void changeImageArray(const unsigned char *images, unsigned char **destImages, const unsigned long width, const unsigned long height, const unsigned long length);

/**
* 将其转换为BMP图片
**/
void toBMPImage(const char* imageFileName, const unsigned char *images, const unsigned long width, const unsigned long height);
#endif