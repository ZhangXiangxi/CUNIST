#ifndef __CUNIST_READDATA_H
#define __CUNIST_READDATA_H

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
int readData(const char* imageFileName, const char* labelFileName, unsigned char *images, unsigned char *labels, int &width, int &height);

#endif