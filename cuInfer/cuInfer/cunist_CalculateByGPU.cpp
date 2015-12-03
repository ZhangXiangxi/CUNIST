#include "cunist_CalculateByGPU.h"
#include <stdlib.h>
#include <time.h>
#include <exception>
/**
	status inference(char * data, char * &result)
	@return	: ���ش����룬���û�д���ͷ���0
	@data	: ͼ�����飬����Ϊ28 * 28 = 784
	@result	: ����һ����ַ�������ַ�д����10��float����Ӧ�˷ֱ�����0~9�ĸ��ʣ���ߵ�һ����Ϊ�жϵĽ��
*/

JNIEXPORT jint JNICALL Java_cunist_CalculateByGPU_inference
(JNIEnv * env, jobject obj, jbyteArray data, jdoubleArray result) {
	void * temp = (env->GetByteArrayElements(data, NULL));
	unsigned char * CData = static_cast<unsigned char*>(temp);
	try{
		for (int i = 0; i < 28 * 28; i++) {
			CData[i] = (CData[i] > 100) ? i / 2 : i;
		}
	} catch (std::exception e) {
		return 1;
	}
	
	double* CResult = env->GetDoubleArrayElements(result, NULL);
	for (int i = 0; i < 10; i++){
		CResult[i] = rand()%8 * 0.1122223333444;
	}
	env->ReleaseDoubleArrayElements(result, CResult, 0);
	return 0;
}