#ifndef __CUNIST_READDATA_H
#define __CUNIST_READDATA_H

#include <string>	// import std::string from string

/**
 *���������ָ�����ļ����ж������ݵ�ָ��ָ��ָ��������У������� MNIST �����ݸ�ʽ
 *�������ָ��Ϊ��ָ�룬��ֻ���س��ȣ�����ȡ����
 *����������󣬷���0
 *
 *@param imageFileName: ͼƬ�ļ����ļ���
 *@param labelFileName: ��ע�ļ����ļ���
 *@param images: ָ�����ݼ����ÿռ��ָ��
 *@param labels: ָ���ע�����ÿռ��ָ��
 *@param width: ͼ��Ŀ��
 *@param height: ͼ��ĸ߶�
 *@return: �������ݵ���Ŀ
 *
*/

int readData(const char* imageFileName, const char* labelFileName, char *images, char *labels, int &width, int &height);

#endif