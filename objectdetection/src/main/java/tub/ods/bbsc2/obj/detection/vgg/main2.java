package tub.ods.bbsc2.obj.detection.vgg;

import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.io.File;
import java.net.URL;

public class main2 {

    private static boolean stop;

    static String result;
    public static void main(String[]args) throws Exception {

            final OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
            File dogImage = loadResource("dog.jpg");
            String video2 = "/home/hasan/bbdc/videoSample2.mp4";

            File flowerImage = loadResource("flower.jpg");
            File alternative = loadResource("alternative.hdf5");
            File alternative1 = loadResource("mobilenet_tf_keras_2.h5");

            //testModel test = new testModel();

           // ImageClassifier ic = new ImageClassifier();
           // final String classify = ic.classify(video2);




        ExFrame();
            //System.out.println(classify);



        }



        public static void ExFrame() throws FrameGrabber.Exception {


            File videoFile = new File("/home/hasan/bbdc/videoSample2.mp4");

            FFmpegFrameGrabber frameGrabber = new FFmpegFrameGrabber(videoFile);
            frameGrabber.start();
            Frame frame;
            for (int i=0;i<50;i++){

                try {

                    frame = frameGrabber.grab();
                    ImageClassifier ic = new ImageClassifier();
                    final String classify = ic.classify(frame);
                    System.out.println(classify);

                } catch (Exception e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }
            frameGrabber.stop();




        }




    private static File loadResource(String s) {
            try{
                URL url = ClassLoader.getSystemResource(s);
                return new File(url.toURI());
            }catch (Exception e){
                throw new RuntimeException(e);
            }
        }
    }


