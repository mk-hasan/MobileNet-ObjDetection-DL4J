package tub.ods.bbdc2.obj.detection;

import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Arrays;

public class RunMobilenet {
    public static void main(String[]args){
        File dogImage = loadResource("dog.jpg");

        File flowerImage = loadResource("flower.jpg");
        File alternative = loadResource("mobilenet_tf_keras_2.h5");

        testModel test = new testModel();
        test.runModel(dogImage,flowerImage,alternative);

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
