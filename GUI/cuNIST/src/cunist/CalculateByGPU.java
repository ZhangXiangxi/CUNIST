package cunist;

/**
 * Created by ZouKaifa on 2015/12/2.
 */
public class CalculateByGPU {
    public native int inference(byte[] data, double[] result);
    static{
        System.loadLibrary("calculate");
    }
    public double[] pros;
    public int get(){
        byte[] d = new byte[10];
        double[] r = new double[10];
        int sta = inference(d, r);
        return sta;
    }
}
