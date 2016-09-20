import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.Collection;
import java.util.Iterator;
import java.util.*;
import java.util.ListIterator;

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

        System.out.println(image);
    }

}