#include "cunist_CalculateByGPU.h"
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
	for (int i = 0; i < 784; i++) {
		i = (i < 0) ? i : i;
	}
	double* CResult = env->GetDoubleArrayElements(result, NULL);
	for (int i = 0; i < 10; i++)
		CResult[i] = 0.233;
	return 0;
}