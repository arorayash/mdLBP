import com.sun.javafx.geom.Vec3d;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.*;

import static org.opencv.core.Core.split;

class HelloCV {

    /** Converting the image into GrayScale **/

    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
        String location = "resources/img0.jpg";
        System.out.println("Coverting the image to grayscale");
        Mat image = Imgcodecs.imread(location);
        List<Mat> mv = new ArrayList<Mat>();
        split(image, mv);
        Mat channel1 = mv.get(0);
        Mat channel2 = mv.get(1);
        Mat channel3 = mv.get(2);

        //Imgproc.cvtColor(channel1, image, Imgproc.COLOR_BGR2GRAY);
        Imgcodecs.imwrite("resources/img0_channel1.jpg", channel1);
        Imgcodecs.imwrite("resources/img0_channel2.jpg", channel2);
        Imgcodecs.imwrite("resources/img0_channel3.jpg", channel3);
        Imgcodecs.imwrite("resources/img0_olbp_chnl1.jpg", olbp(channel1));
        Imgcodecs.imwrite("resources/img0_olbp_chnl2.jpg", olbp(channel2));
        Imgcodecs.imwrite("resources/img0_olbp_chnl3.jpg", olbp(channel3));
        System.out.println(image);
    }

    public static Mat olbp( Mat channel )
    {
        Mat dest = new Mat(channel.rows(),channel.cols(), channel.channels());

        for (int i = 1; i < channel.rows()-1 ; i++)
        {
            for (int j = 1; j < channel.cols()-1; j++)
            {
                double[] center = channel.get(i,j);
                int x = (int)center[0];
                int[] arr = new int[8];

                if (channel.get(i-1,j-1)[0]>x)
                    arr[0] = 1;
                else
                    arr[0] = 0;

                if(channel.get(i-1,j)[0]>x)
                    arr[1] = 0;
                else
                    arr[1] = 1;

                if(channel.get(i-1,j+1)[0]>x)
                    arr[2] = 0;
                else
                    arr[2] = 1;

                if(channel.get(i,j+1)[0]>x)
                    arr[3] = 0;
                else
                    arr[3] = 1;

                if(channel.get(i+1,j+1)[0]>x)
                    arr[4] = 0;
                else
                    arr[4] = 1;

                if(channel.get(i, j+1)[0]>x)
                    arr[5] = 0;
                else
                    arr[5] = 1;

                if(channel.get(i-1, j+1)[0]>x)
                    arr[6] = 0;
                else
                    arr[6] = 1;

                if(channel.get(i, j-1)[0]>x)
                    arr[7] = 0;
                else
                    arr[7] = 1;

                int dec = bin2dec(arr);

                dest.put(i, j, dec);
                //code |= (channel.get(i-1,j-1)) << 7;
                //String s = Arrays.toString(center);
                //System.out.println((int)Double.parseDouble(s.substring(1,s.length()-1)));
                //System.out.println(dec + " " + dest.get(i,j));
            }
        }
        //System.out.println(dest);
        //Imgcodecs.imwrite("resources/img0_try.jpg", dest);
        return dest;
    }

    public static int bin2dec(int arr[])
    {
        int result = 0;
        for(int i=0; i<arr.length; i++)
        {
            result = result + arr[i]*((int)Math.pow(2,i));
        }
        return result;
    }

}