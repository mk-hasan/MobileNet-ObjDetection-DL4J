package tub.ods.bbdc2.obj.detection;

import java.io.File;
import java.net.URL;

public class RunMobilenet {
    public static void main(String[]args){
        File dogImage = loadResource("dog.jpg");

        File flowerImage = loadResource("flower.jpg");
        File alternative = loadResource("alternative.hdf5");
        File alternative1 = loadResource("mobilenet_tf_keras_2.h5");

        //testModel test = new testModel();


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
