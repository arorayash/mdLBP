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
        System.out.println("Image is split into three channels.");

        //Imgproc.cvtColor(channel1, image, Imgproc.COLOR_BGR2GRAY);
        Imgcodecs.imwrite("resources/img0_channel1.jpg", channel1);
        Imgcodecs.imwrite("resources/img0_channel2.jpg", channel2);
        Imgcodecs.imwrite("resources/img0_channel3.jpg", channel3);

        Imgcodecs.imwrite("resources/img0_olbp_chnl1.jpg", olbp(channel1));
        Imgcodecs.imwrite("resources/img0_olbp_chnl2.jpg", olbp(channel2));
        Imgcodecs.imwrite("resources/img0_olbp_chnl3.jpg", olbp(channel3));

        int[] fn = weighingfunc();
        Mat malbp1 = new Mat(channel1.rows(),channel1.cols(), channel1.channels());
        Mat malbp2 = new Mat(channel1.rows(),channel1.cols(), channel1.channels());
        Mat malbp3 = new Mat(channel1.rows(),channel1.cols(), channel1.channels());
        Mat malbp4 = new Mat(channel1.rows(),channel1.cols(), channel1.channels());

        Mat mdlbp1 = new Mat(channel1.rows(),channel1.cols(), channel1.channels());
        Mat mdlbp2 = new Mat(channel1.rows(),channel1.cols(), channel1.channels());
        Mat mdlbp3 = new Mat(channel1.rows(),channel1.cols(), channel1.channels());
        Mat mdlbp4 = new Mat(channel1.rows(),channel1.cols(), channel1.channels());
        Mat mdlbp5 = new Mat(channel1.rows(),channel1.cols(), channel1.channels());
        Mat mdlbp6 = new Mat(channel1.rows(),channel1.cols(), channel1.channels());
        Mat mdlbp7 = new Mat(channel1.rows(),channel1.cols(), channel1.channels());
        Mat mdlbp8 = new Mat(channel1.rows(),channel1.cols(), channel1.channels());

        for (int i = 1; i < channel1.rows()-1 ; i++)
        {
            for (int j = 1; j < channel1.cols() - 1; j++)
            {
                double[] center1 = channel1.get(i,j);
                int x1 = (int)center1[0];
                double[] center2 = channel2.get(i,j);
                int x2 = (int)center2[0];
                double[] center3 = channel3.get(i,j);
                int x3 = (int)center3[0];
                int[] lbparr1 = lbpArray(channel1,i, j, x1);
                int[] lbparr2 = lbpArray(channel2,i, j, x2);
                int[] lbparr3 = lbpArray(channel3,i, j, x3);

                int[] mam = mam(lbparr1, lbparr2, lbparr3);
                int[] mdm = mdm(lbparr1, lbparr2, lbparr3);

                int[] malbptn1 = malbpn(mam, 0);
                int[] malbptn2 = malbpn(mam, 1);
                int[] malbptn3 = malbpn(mam, 2);
                int[] malbptn4 = malbpn(mam, 3);

                int malbpt1 = malbp(malbptn1, fn);
                int malbpt2 = malbp(malbptn2, fn);
                int malbpt3 = malbp(malbptn3, fn);
                int malbpt4 = malbp(malbptn4, fn);

                int[][] mdlbptn = mdlbptn(mdm);
                int[] mdlbpt = mdlbpt(mdlbptn, fn);

                malbp1.put(i, j, malbpt1);
                malbp2.put(i, j, malbpt2);
                malbp3.put(i, j, malbpt3);
                malbp4.put(i, j, malbpt4);

                mdlbp1.put(i, j, mdlbpt[0]);
                mdlbp2.put(i, j, mdlbpt[1]);
                mdlbp3.put(i, j, mdlbpt[2]);
                mdlbp4.put(i, j, mdlbpt[3]);
                mdlbp5.put(i, j, mdlbpt[4]);
                mdlbp6.put(i, j, mdlbpt[5]);
                mdlbp7.put(i, j, mdlbpt[6]);
                mdlbp8.put(i, j, mdlbpt[7]);

                //System.out.println(lbparr1[7]);
            }
        }

        //Mat mam = madder(channel1, channel2, channel3);

      //  System.out.println("maM : \n"+mam);
        Imgcodecs.imwrite("resources/img0_malbp1.jpg", malbp1);
        Imgcodecs.imwrite("resources/img0_malbp2.jpg", malbp2);
        Imgcodecs.imwrite("resources/img0_malbp3.jpg", malbp3);
        Imgcodecs.imwrite("resources/img0_malbp4.jpg", malbp4);

        Imgcodecs.imwrite("resources/img0_mdlbp1.jpg", mdlbp1);
        Imgcodecs.imwrite("resources/img0_mdlbp2.jpg", mdlbp2);
        Imgcodecs.imwrite("resources/img0_mdlbp3.jpg", mdlbp3);
        Imgcodecs.imwrite("resources/img0_mdlbp4.jpg", mdlbp4);
        Imgcodecs.imwrite("resources/img0_mdlbp5.jpg", mdlbp5);
        Imgcodecs.imwrite("resources/img0_mdlbp6.jpg", mdlbp6);
        Imgcodecs.imwrite("resources/img0_mdlbp7.jpg", mdlbp7);
        Imgcodecs.imwrite("resources/img0_mdlbp8.jpg", mdlbp8);

        //maLBP(mam, 1);

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
                int[] lbparr = lbpArray(channel,i, j, x);
                int dec = bin2dec(lbparr);

                dest.put(i, j, dec);
                //code |= (channel.get(i-1,j-1)) << 7;
                //String s = Arrays.toString(center);
                //System.out.println((int)Double.parseDouble(s.substring(1,s.length()-1)));
                //System.out.println(dec + " " + dest.get(i,j));
            }
        }
        //System.out.println(dest);
        //Imgcodecs.imwrite("resources/img0_try.jpg", dest);
        System.out.println(dest);
        return dest;
    }

    public static int[] lbpArray( Mat channel, int locX, int locY, int center )
    {
        int[] arr = new int[8];

        if (channel.get(locX-1,locY-1)[0]>center)
            arr[0] = 1;
        else
            arr[0] = 0;

        if(channel.get(locX-1,locY)[0]>center)
            arr[1] = 0;
        else
            arr[1] = 1;

        if(channel.get(locX-1,locY+1)[0]>center)
            arr[2] = 0;
        else
            arr[2] = 1;

        if(channel.get(locX,locY+1)[0]>center)
            arr[3] = 0;
        else
            arr[3] = 1;

        if(channel.get(locX+1,locY+1)[0]>center)
            arr[4] = 0;
        else
            arr[4] = 1;

        if(channel.get(locX, locY+1)[0]>center)
            arr[5] = 0;
        else
            arr[5] = 1;

        if(channel.get(locX-1, locY+1)[0]>center)
            arr[6] = 0;
        else
            arr[6] = 1;

        if(channel.get(locX, locY-1)[0]>center)
            arr[7] = 0;
        else
            arr[7] = 1;

        return arr;
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

    public static int[] mam(int[] a, int[] b, int[] c)
    {
        int[] mam = new int[8];
        for(int i = 0; i<a.length; i++)
        {
            mam[i] = a[i]+b[i]+c[i];
        }
        return mam;
    }

    private static int[] mdm(int[] lbparr1, int[] lbparr2, int[] lbparr3)
    {
        int[] mdmArr = new int[8];
        for (int i = 0; i<8; i++)
        {
            mdmArr[i] = 4*lbparr1[i] + 2*lbparr2[i] + lbparr3[i];
        }
        return mdmArr;
    }

    private static int[] malbpn(int[] mam, int t) {
        //t denotes the output channel number
        int[] malbpnArr = new int[8];
        for(int i = 0; i<8; i++)
        {
            if(mam[i]==t)
                malbpnArr[i] = 1;
            else
                malbpnArr[i] = 0;
        }
        return malbpnArr;
    }

    private static int[][] mdlbptn( int[] mdm )
    {
        int[][] mdlbptnArr = new int[8][8];
        int t2 = 1;  //t2 belongs to [1, 2^c]
        for(int i=0; i<8; i++)
        {
            for(int j=0; j<8; j++)
            {
                if(mdm[j]==t2-1)
                    mdlbptnArr[i][j] = 1;
                else
                    mdlbptnArr[i][j] = 0;
            }
            t2++;
        }
        return mdlbptnArr;
    }

    public static int[] weighingfunc()
    {
        int[] fn = new int[8];
        for(int i=0; i<8; i++)
        {
            fn[i] = (int) Math.pow(2,i-1);
        }
        return fn;
    }

    /*public static int[] lbptn( Mat c1, Mat c2, Mat c3)
    {

       Mat temp = new Mat(c1.rows(), c2.cols(), 1);
        Core.add(c1, c2, temp);
        Core.add(c3, temp, temp);
        return  temp;

    }
    */

    public static int malbp(int[] malbpn, int[] fn)
    {
        int malbpVal = 0;
        for(int i=0; i<8; i++)
        {
            malbpVal = malbpVal + malbpn[i]*fn[i];
        }
        return malbpVal;
    }

    public static int[] mdlbpt(int[][] mdlbptn, int[] fn)
    {
        int[] mdlbpVals = new int[8];
        for(int i=0; i<8; i++)
        {
            int mdlbpVal = 0;
            for(int j=0; j<8; j++)
            {
                mdlbpVal = mdlbpVal + mdlbptn[i][j]*fn[i];
            }
            mdlbpVals[i] = mdlbpVal;
        }
        return mdlbpVals;

    }

    public static Mat maLBP( Mat mam, int t)
    {
        //t belongs to [1, c+1]
        Mat maLBP = new Mat(mam.rows(), mam.cols(), 1);

        Size len = maLBP.size();
        System.out.println("malbp size: "+ len);

        for(int i=0; i<(int)len.height ; i++)
        {
            for(int j=0; j<(int)len.width; j++)
            {
                double[] a = mam.get(i,j);
               // System.out.println(a[0]);
            }
        }
        return maLBP;
    }

}