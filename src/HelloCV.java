import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class HelloCV {

    /** Converting the image into GrayScale **/

    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
        String location = "resources/img0.jpg";
        System.out.println("Coverting the image to grayscale");
        Mat image = Imgcodecs.imread(location);
        Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2GRAY);
        Imgcodecs.imwrite("resources/img0_gray.jpg", image);
        System.out.println("Done.");
    }

}