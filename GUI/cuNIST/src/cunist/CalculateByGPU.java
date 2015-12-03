package cunist;

/**
 * Created by ZouKaifa on 2015/12/2.
 */
public class CalculateByGPU {
    public native int inference(float[] data, double[] result);
    static{
        System.loadLibrary("cuInfer");
    }
    public int get(float[] b, double[] pros){
        int sta = inference(b, pros);
        return sta;
    }
}
