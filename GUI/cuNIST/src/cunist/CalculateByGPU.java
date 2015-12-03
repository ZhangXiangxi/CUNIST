package cunist;

/**
 * Created by ZouKaifa on 2015/12/2.
 */
public class CalculateByGPU {
    public native int inference(int[] data, double[] result);
    static{
        System.loadLibrary("cuInfer");
    }
    public int get(int[] b, double[] pros){
        int sta = inference(b, pros);
        return sta;
    }
}
