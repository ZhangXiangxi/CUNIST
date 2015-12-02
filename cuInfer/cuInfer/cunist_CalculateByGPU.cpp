#include "cunist_CalculateByGPU.h"
/**
	status inference(char * data, char * &result)
	@return	: 返回错误码，如果没有错误就返回0
	@data	: 图像数组，长度为28 * 28 = 784
	@result	: 返回一个地址，这个地址中存放了10个float，对应了分别属于0~9的概率，最高的一个作为判断的结果
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