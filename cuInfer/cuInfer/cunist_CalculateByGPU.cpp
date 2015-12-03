#include "cunist_CalculateByGPU.h"
#include <stdlib.h>
#include <time.h>
#include <exception>
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